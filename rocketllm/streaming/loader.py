"""Reading a module's weights off storage into one contiguous host buffer.

Three things make this cheaper than the obvious loop over ``get_tensor``.

The first is that a tensor is read by its byte range rather than by mapping the file it lives in.
Reading one MoE expert costs that expert's bytes and not the whole layer file -- for a large mixture
that is the difference between a few megabytes and sixteen gigabytes -- and the range read is what
lets a machine that charges for mappings stream a checkpoint far larger than its commit limit. That
is :mod:`rocketllm.streaming.shards`, which owns everything about how a shard is read.

The second is that everything lands in slices of a *single* host buffer, laid out end to end. That
buffer is what crosses the link, in one transfer, instead of sixty small ones that never reach the
link's rated bandwidth and pay per-transfer overhead sixty times.

The third is that reads run on a small thread pool whose width is ``HardwareProfile.io_workers`` --
the concurrency that was *measured* to saturate this machine's storage. A single reader does not
reach a fast NVMe's rated bandwidth because it is latency-bound rather than bandwidth-bound; too
many readers make a slow or rotational device thrash. There is no correct constant for that, which
is exactly why it is probed. Each worker gets a contiguous stretch of the shard, so widening the
pool splits one sequential sweep into several rather than turning it into a scatter of seeks.
"""
import dataclasses
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from . import shards
from .shards import torch_dtype_of  # noqa: F401 - part of this module's published surface

log = logging.getLogger(__name__)

#: Byte alignment for every tensor's slot in the shared buffer. Reinterpreting a slice of a uint8
#: buffer as another dtype requires the offset to be a multiple of that dtype's item size, and 16
#: covers every dtype torch has, including anything wider added later.
ALIGNMENT = 16


def _align_up(value, to=ALIGNMENT):
    return (value + to - 1) // to * to


@dataclasses.dataclass(frozen=True)
class TensorPlacement:
    """Where one checkpoint tensor sits inside the shared host buffer."""

    name: str
    offset: int
    nbytes: int
    shape: tuple
    dtype: object
    #: Row range to read, for a fused expert tensor whose rows are separate experts.
    row_start: object = None
    row_stop: object = None

    @property
    def sliced(self):
        return self.row_start is not None

    def raw_view_into(self, buffer):
        """This tensor's slot in the shared buffer, still as bytes. What a read writes into."""
        return buffer.narrow(0, self.offset, self.nbytes)

    def view_into(self, buffer):
        """This tensor, as a view of the shared buffer. No copy."""
        return self.raw_view_into(buffer).view(self.dtype).view(self.shape)


@dataclasses.dataclass(frozen=True)
class LayerLayout:
    """Every tensor a module needs, and the single buffer they will be packed into."""

    layer_name: str
    placements: tuple
    total_bytes: int

    def views(self, buffer):
        return {p.name: p.view_into(buffer) for p in self.placements}


class LoadedLayer:
    """A staged layer: the host buffer, what is in it, and the lease that owns it.

    The lease is the part that matters. It must outlive the transfer reading from this buffer, so
    it is carried here rather than released when loading finishes.
    """

    __slots__ = ("layout", "lease", "layer_name")

    def __init__(self, layout, lease):
        self.layout = layout
        self.lease = lease
        self.layer_name = layout.layer_name

    @property
    def buffer(self):
        return self.lease.view

    @property
    def nbytes(self):
        return self.layout.total_bytes

    def views(self):
        return self.layout.views(self.buffer)

    def release(self):
        self.lease.release()


class LayerLoader:
    """Reads modules out of the split checkpoint into pooled host buffers.

    It owns no file state of its own. Everything about opening a shard, parsing its header and
    fetching a byte range belongs to the checkpoint's :class:`~rocketllm.streaming.shards.
    ShardReader`, which every other reader in the process shares -- the free functions in
    ``rocketllm.utils`` that the cache's prefetch workers call among them. That sharing is not
    tidiness. Whatever limit the machine imposes has to hold across the process, and a limit each
    call site applied to itself would be several limits and no ceiling.
    """

    #: Handle-limit settings other than a plain count. Re-exported from the reader that honours
    #: them so callers have one name to use.
    AUTO = shards.ShardReader.AUTO
    UNBOUNDED = shards.ShardReader.UNBOUNDED

    def __init__(self, checkpoint_path, pool, profile=None, io_workers=None,
                 shard_handle_limit=AUTO):
        self.checkpoint_path = Path(checkpoint_path)
        self.pool = pool
        self.io_workers = self._resolve_workers(profile, io_workers)
        self._executor = (ThreadPoolExecutor(max_workers=self.io_workers,
                                             thread_name_prefix="rocketllm-io")
                          if self.io_workers > 1 else None)
        # The engine's settings, not a free function's guess, decide how this checkpoint is read
        # from here on -- so this replaces any reader an earlier bare call already registered.
        self.reader = shards.configure_reader(
            self.checkpoint_path, profile=profile, shard_handle_limit=shard_handle_limit,
            # The loader's own pool is not the only thing reading: the cache prefetches on a pool
            # of its own width, and the thread running the forward reads too.
            readers=self.io_workers * 2 + 1)
        self.reads = 0
        self.bytes_read = 0

    @staticmethod
    def _resolve_workers(profile, override):
        if override is not None:
            return max(1, int(override))
        if profile is not None:
            derivation = profile.derived.get("io_workers")
            if derivation is not None:
                return max(1, int(derivation.value))
        # One reader is the only width that is correct everywhere; it is simply not the fastest.
        return 1

    def shard_path(self, layer_name):
        return self.checkpoint_path / (layer_name + ".safetensors")

    # -- planning ------------------------------------------------------------------------------

    def plan(self, layer_name, keys=None, rows=None):
        """Lay the requested tensors out end to end, reading only the shard's header.

        Nothing is read here beyond shape and dtype, which is what lets the cache size a layer in
        packed bytes before deciding whether it wants it -- and, since the header is all that is
        touched, lets it size a checkpoint far larger than the machine could ever hold.
        """
        index = self.reader.index(self.shard_path(layer_name))
        names = list(keys) if keys is not None else index.keys()
        rows = rows or {}

        placements = []
        offset = 0
        for name in names:
            entry = index[name]
            shape = entry.shape
            row_start = row_stop = None
            if name in rows:
                row_start, row_stop = rows[name]
                shape = (int(row_stop) - int(row_start),) + shape[1:]
            count = 1
            for dim in shape:
                count *= dim
            nbytes = count * torch.empty((), dtype=entry.dtype).element_size()
            placements.append(TensorPlacement(name=name, offset=offset, nbytes=nbytes, shape=shape,
                                              dtype=entry.dtype, row_start=row_start,
                                              row_stop=row_stop))
            offset = _align_up(offset + nbytes)

        return LayerLayout(layer_name=layer_name, placements=tuple(placements), total_bytes=offset)

    # -- reading -------------------------------------------------------------------------------

    def load(self, layer_name, keys=None, rows=None, layout=None):
        """Read a module into one leased host buffer and return it staged, not yet transferred."""
        layout = layout if layout is not None else self.plan(layer_name, keys, rows)
        lease = self.pool.lease(layout.total_bytes, torch.uint8)
        try:
            self._fill(layout, lease.view)
        except BaseException:
            # A half-filled buffer must not be left checked out, or the pool leaks a slot for the
            # rest of the run.
            lease.release()
            raise
        return LoadedLayer(layout, lease)

    def _fill(self, layout, buffer):
        path = self.shard_path(layout.layer_name)
        # Each placement owns a disjoint slice of the buffer, so the readers never overlap and the
        # destination needs no locking.
        requests = [shards.ReadRequest(name=p.name, destination=p.raw_view_into(buffer),
                                       row_start=p.row_start, row_stop=p.row_stop)
                    for p in layout.placements]
        if not requests:
            return
        self.reads += len(requests)
        self.bytes_read += sum(p.nbytes for p in layout.placements)

        if self._executor is None or len(requests) < 2:
            self.reader.read_into(path, requests)
            return
        batches = self.reader.partition(path, requests, self.io_workers)
        if len(batches) < 2:
            self.reader.read_into(path, batches[0])
            return
        futures = [self._executor.submit(self.reader.read_into, path, batch)
                   for batch in batches[1:]]
        try:
            # The submitting thread takes the first batch rather than waiting on the pool for all
            # of them: with io_workers of 2 that is the difference between one reader and two.
            self.reader.read_into(path, batches[0])
        finally:
            errors = [future.exception() for future in futures]
        for error in errors:
            if error is not None:
                raise error

    # -- lifecycle -----------------------------------------------------------------------------

    def stats(self):
        stats = {
            "io_workers": self.io_workers,
            "reads": self.reads,
            "bytes_read": self.bytes_read,
        }
        stats.update(self.reader.stats())
        return stats

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        # Anything the reader holds open keeps the shard alive on the filesystem: on Windows a
        # mapped file cannot be deleted, so waiting for the collector to notice is not good enough.
        shards.release_reader(self.checkpoint_path, only=self.reader)
        self.reader.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
