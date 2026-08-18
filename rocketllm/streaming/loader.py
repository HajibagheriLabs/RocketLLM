"""Reading a module's weights off storage into one contiguous host buffer.

Two things make this cheaper than the obvious loop over ``get_tensor``.

The first is ``get_slice``. safetensors can seek to a byte range inside a shard, so reading one MoE
expert costs that expert's bytes and not the whole layer file -- for a large mixture that is the
difference between a few megabytes and sixteen gigabytes. The same mechanism reads one row out of a
fused 3D expert tensor.

The second is that everything lands in slices of a *single* host buffer, laid out end to end. That
buffer is what crosses the link, in one transfer, instead of sixty small ones that never reach the
link's rated bandwidth and pay per-transfer overhead sixty times.

Reads run on a small thread pool whose width is ``HardwareProfile.io_workers`` -- the concurrency
that was *measured* to saturate this machine's storage. A single reader does not reach a fast NVMe's
rated bandwidth because it is latency-bound rather than bandwidth-bound; too many readers make a
slow or rotational device thrash. There is no correct constant for that, which is exactly why it is
probed.
"""
import collections
import dataclasses
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open

from ..hw import caps

log = logging.getLogger(__name__)

#: safetensors dtype codes to torch dtypes. Codes, not sizes: the file says what it holds.
_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
}
for _name, _code in (("float8_e4m3fn", "F8_E4M3"), ("float8_e5m2", "F8_E5M2"),
                     ("uint16", "U16"), ("uint32", "U32"), ("uint64", "U64")):
    _dtype = getattr(torch, _name, None)
    if _dtype is not None:
        _DTYPES[_code] = _dtype

#: Byte alignment for every tensor's slot in the shared buffer. Reinterpreting a slice of a uint8
#: buffer as another dtype requires the offset to be a multiple of that dtype's item size, and 16
#: covers every dtype torch has, including anything wider added later.
ALIGNMENT = 16


def torch_dtype_of(code):
    dtype = _DTYPES.get(str(code).upper())
    if dtype is None:
        raise ValueError(f"safetensors dtype {code!r} is not one this build of torch can represent")
    return dtype


def _align_up(value, to=ALIGNMENT):
    return (value + to - 1) // to * to


def _close_handle(handle):
    """Unmap a shard handle now, rather than whenever the collector gets to it.

    safetensors exposes the release through the context-manager protocol and nothing else, so that
    is what is called. Measured on Windows: twenty shards totalling 50GB took 7.7GB of commit while
    open and gave 50GB back the moment this ran, which is the whole mechanism this file relies on.
    """
    closer = getattr(handle, "__exit__", None)
    if closer is None:
        return
    try:
        closer(None, None, None)
    except Exception:  # noqa: BLE001 - a handle that will not close must not fail a read
        log.debug("a shard handle did not close cleanly", exc_info=True)


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

    def view_into(self, buffer):
        """This tensor, as a view of the shared buffer. No copy."""
        flat = buffer.narrow(0, self.offset, self.nbytes)
        return flat.view(self.dtype).view(self.shape)


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

    One open ``safe_open`` handle per shard used to be kept for the loader's lifetime, on the
    reasoning that a handle is cheap. It is cheap in file descriptors and it is not cheap in
    address space: a handle holds the shard MEMORY-MAPPED, and an OS that charges mappings against
    a commit limit is charged the shard's full size the moment it is opened, touched or not. Sizing
    a 67GB checkpoint reads every shard's header once, which meant mapping all of it at once -- so
    the engine died in ``safe_open`` on a machine with an 18GB commit limit while doing nothing but
    reading headers, having never come close to running out of the memory it actually uses.

    So the cache is bounded when, and only when, the machine says it has to be. See
    :meth:`_resolve_handle_limit`: the budget comes from measured commit headroom via the hardware
    profile, the footprint from this checkpoint's own shards, and where the two allow it every
    handle stays open exactly as before.
    """

    #: Handle-limit settings other than a plain count.
    AUTO = "auto"
    UNBOUNDED = "unbounded"

    def __init__(self, checkpoint_path, pool, profile=None, io_workers=None,
                 shard_handle_limit=AUTO):
        self.checkpoint_path = Path(checkpoint_path)
        self.pool = pool
        self.io_workers = self._resolve_workers(profile, io_workers)
        self._executor = (ThreadPoolExecutor(max_workers=self.io_workers,
                                             thread_name_prefix="rocketllm-io")
                          if self.io_workers > 1 else None)
        self.handle_limit, self.handle_mode, self.handle_reason = self._resolve_handle_limit(
            profile, shard_handle_limit)
        #: Handles per reader thread. Zero means unbounded; otherwise the total is shared out, so
        #: the bound holds across the pool rather than being applied to each worker separately.
        self._per_thread_limit = (0 if self.handle_limit <= 0
                                  else max(1, self.handle_limit // max(1, self.io_workers)))
        if self._per_thread_limit:
            # A reader cannot hold fewer than the one handle it is reading through, so a limit
            # below the worker count cannot be met. Publish what will actually be held rather than
            # what was asked for: this number goes into the report, and a bound that is quietly
            # exceeded is worse than one that is honestly larger.
            effective = self._per_thread_limit * self.io_workers
            if effective != self.handle_limit:
                self.handle_reason += (f" (raised from {self.handle_limit} to {effective}: "
                                       f"{self.io_workers} readers cannot share fewer)")
                self.handle_limit = effective
        # safetensors handles are cheap but not documented as thread-safe, so each worker opens its
        # own rather than sharing one across the pool. Every one of them is also recorded centrally,
        # because thread-local state is not reachable from the thread that shuts the loader down.
        self._local = threading.local()
        self._handles = set()
        self._handles_lock = threading.Lock()
        self.reads = 0
        self.bytes_read = 0
        self.handle_opens = 0
        self.handle_evictions = 0
        if self.handle_mode == "bounded":
            caps.announce_once(
                "shard-handle-bound",
                f"holding at most {self.handle_limit} shard file(s) mapped at a time "
                f"({self._per_thread_limit} per reader): {self.handle_reason}. Reads are "
                f"unaffected; a shard is reopened when it comes round again.", logging.INFO)

    # -- how many shards may stay mapped ---------------------------------------------------------

    def checkpoint_bytes(self):
        """What this checkpoint would cost to hold mapped in its entirety."""
        try:
            return sum(path.stat().st_size for path in self.checkpoint_path.glob("*.safetensors"))
        except OSError:
            return 0

    def largest_shard_bytes(self):
        try:
            return max((path.stat().st_size
                        for path in self.checkpoint_path.glob("*.safetensors")), default=0)
        except OSError:
            return 0

    def _resolve_handle_limit(self, profile, setting):
        """``(limit, mode, why)``. A limit of 0 means every handle stays open.

        The decision is a comparison of two measured numbers -- what the checkpoint would cost to
        keep mapped, against what this machine will let the process charge -- and never a platform
        test. A machine that does not charge for mappings publishes no budget, so this returns
        unbounded there without knowing or caring which machine it is.
        """
        if isinstance(setting, str):
            if setting == self.UNBOUNDED:
                return 0, "unbounded", "requested explicitly"
            if setting != self.AUTO:
                raise ValueError(
                    f"shard_handle_limit must be {self.AUTO!r}, {self.UNBOUNDED!r} or an integer, "
                    f"not {setting!r}")
        elif setting is not None:
            limit = int(setting)
            if limit <= 0:
                return 0, "unbounded", "requested explicitly"
            return limit, "bounded", f"requested explicitly ({limit})"

        budget = 0
        floor, ceiling = 2, 4
        if profile is not None:
            derivation = profile.derived.get("shard_mapping_budget_bytes")
            if derivation is not None:
                budget = max(0, int(derivation.value))
            inputs = getattr(profile.derived.get("shard_handle_limit"), "inputs", None) or {}
            floor = int(inputs.get("shard_handle_floor", floor))
            ceiling = int(inputs.get("shard_handle_ceiling", ceiling))
        if budget <= 0:
            return 0, "unbounded", ("this machine does not charge memory mappings against a commit "
                                    "limit, or it could not be measured")

        footprint = self.checkpoint_bytes()
        if footprint <= budget:
            # Includes a footprint of zero, which means the shards could not be measured or are not
            # there yet. Bounding on the strength of a number we do not have would trade a real
            # cost -- reopening a shard per layer -- for a risk we have no evidence of.
            return 0, "unbounded", (f"the whole checkpoint ({footprint / 1024 ** 3:.1f}GB) fits in "
                                    f"the {budget / 1024 ** 3:.1f}GB that may be held mapped")

        largest = self.largest_shard_bytes()
        affordable = int(budget // largest) if largest else ceiling
        limit = max(floor, min(ceiling, affordable))
        return limit, "bounded", (
            f"the checkpoint would cost {footprint / 1024 ** 3:.1f}GB held mapped against a "
            f"{budget / 1024 ** 3:.1f}GB budget, and the largest shard is "
            f"{largest / 1024 ** 3:.2f}GB")

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

    def _open(self, layer_name):
        """A handle for this shard, opening it if the calling thread does not already hold one.

        Least-recently-used order, and eviction closes the handle rather than dropping the
        reference: an unmapped section is what gives the commit charge back, and waiting for the
        collector to notice defeats the point of bounding this at all.

        Per thread, never shared. Evicting is therefore safe without any coordination -- a handle
        can only be evicted by the thread that owns it, which is not inside a read when it does so.
        Every read copies out of the mapping before returning, so nothing outlives the handle.
        """
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = self._local.handles = collections.OrderedDict()
        path = str(self.shard_path(layer_name))

        handle = handles.get(path)
        if handle is not None:
            handles.move_to_end(path)
            return handle

        handle = safe_open(path, framework="pt")
        handles[path] = handle
        with self._handles_lock:
            self._handles.add(handle)
            self.handle_opens += 1

        while self._per_thread_limit and len(handles) > self._per_thread_limit:
            _, evicted = handles.popitem(last=False)
            with self._handles_lock:
                self._handles.discard(evicted)
                self.handle_evictions += 1
            _close_handle(evicted)
        return handle

    # -- planning ------------------------------------------------------------------------------

    def plan(self, layer_name, keys=None, rows=None):
        """Lay the requested tensors out end to end, reading only the shard's header.

        Nothing is read here beyond shape and dtype, which is what lets the cache size a layer in
        packed bytes before deciding whether it wants it.
        """
        handle = self._open(layer_name)
        names = list(keys) if keys is not None else list(handle.keys())
        rows = rows or {}

        placements = []
        offset = 0
        for name in names:
            entry = handle.get_slice(name)
            shape = tuple(int(d) for d in entry.get_shape())
            dtype = torch_dtype_of(entry.get_dtype())
            row_start = row_stop = None
            if name in rows:
                row_start, row_stop = rows[name]
                shape = (int(row_stop) - int(row_start),) + shape[1:]
            count = 1
            for dim in shape:
                count *= dim
            nbytes = count * torch.empty((), dtype=dtype).element_size()
            placements.append(TensorPlacement(name=name, offset=offset, nbytes=nbytes, shape=shape,
                                              dtype=dtype, row_start=row_start, row_stop=row_stop))
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
        if self._executor is None or len(layout.placements) < 2:
            for placement in layout.placements:
                self._read_one(layout.layer_name, placement, buffer)
            return
        # Each placement owns a disjoint slice of the buffer, so the workers never overlap and no
        # locking is needed on the destination.
        futures = [self._executor.submit(self._read_one, layout.layer_name, placement, buffer)
                   for placement in layout.placements]
        for future in futures:
            future.result()

    def _read_one(self, layer_name, placement, buffer):
        entry = self._open(layer_name).get_slice(placement.name)
        if placement.sliced:
            # Only the routed rows. One expert costs its own bytes, not the layer's.
            source = entry[placement.row_start:placement.row_stop]
        else:
            source = entry[:]
        destination = placement.view_into(buffer)
        destination.copy_(source)
        self.reads += 1
        self.bytes_read += placement.nbytes

    # -- lifecycle -----------------------------------------------------------------------------

    def stats(self):
        return {
            "io_workers": self.io_workers,
            "reads": self.reads,
            "bytes_read": self.bytes_read,
            # Reported because a bounded run reopens shards, and "why is this slower than the same
            # model on the other box" is answered by these three numbers and nothing else.
            "shard_handle_mode": self.handle_mode,
            "shard_handle_limit": self.handle_limit,
            "shard_handle_opens": self.handle_opens,
            "shard_handle_evictions": self.handle_evictions,
        }

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        # One open handle is kept per shard per reader thread, so a deep model on a wide pool holds
        # hundreds of them at once -- enough to reach a default descriptor limit. They also keep the
        # shard mapped, which on Windows means the file cannot be deleted while the loader lives.
        # Waiting for the garbage collector to notice is not good enough for either.
        with self._handles_lock:
            handles, self._handles = self._handles, set()
        for handle in handles:
            _close_handle(handle)
        self._local = threading.local()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
