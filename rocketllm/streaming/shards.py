"""Reading tensors out of safetensors shards, by whichever route this machine can afford.

There are two, and which is cheaper is a property of the host rather than of the code.

Mapping the file is normally the better one. ``safe_open`` hands back tensors that ALIAS the
mapping instead of copying out of it -- measured: every tensor in a 1.57GB shard came back pointing
into a single 1.57GB region, in four milliseconds -- so the read costs no copy at all and a second
read of the same shard is served from the page cache.

The catch is that same aliasing, on a system that charges mappings against a commit limit. Windows
does: opening a handle to a 1.57GB shard consumed 1.57GB of commit before a byte was read, and a
tensor aliasing it keeps that charge alive for as long as the tensor does, whatever the handle
does. One cached 4MB expert therefore held its whole shard, and closing handles could not release
what live tensors were still pointing at. That is how a 67GB checkpoint exhausted an 18GB commit
limit, and why Windows then reported an unbackable mapped page as an access violation rather than
as a failed allocation -- the engine did not raise, it died. Linux charges a read-only file mapping
nothing at all under its default overcommit policy, which is why this went unseen for so long.

The other route is to read the byte range. The safetensors container makes that easy: eight
little-endian length bytes, that many bytes of JSON naming every tensor's dtype, shape and byte
range, then the tensor data. Nothing about it is version- or architecture-specific. It costs one
copy into a buffer the engine already owns and holds nothing mapped, so a checkpoint of any size
runs against any commit limit.

So the reader measures and picks: map where the checkpoint fits what the machine will let the
process charge, read ranges where it does not. Headers are always read directly, which is what lets
the engine size a 67GB model without charging a byte for it. A big-endian host cannot use the
container's payload as stored, so it maps whatever happens, and there the handle bound is what is
left to keep it inside the limit.

The caller gets the same bytes either way; the tests check both against ``safe_open`` tensor for
tensor, on every dtype this build of torch has.
"""
import collections
import dataclasses
import json
import logging
import os
import struct
import sys
import threading

import torch

from ..hw import caps

log = logging.getLogger(__name__)

#: safetensors dtype codes to torch dtypes. Codes, not sizes: the file says what it holds.
#:
#: Seeded from safetensors' own table where it has one, so this cannot fall behind the reader it is
#: replacing when a new float format lands -- which for this project is not hypothetical, since the
#: quantized checkpoints it exists to run are exactly where new formats appear first. The literals
#: are the floor for a build that does not expose the table, and an entry only survives if torch can
#: actually size it: a sub-byte format whose elements do not divide into bytes has no place in a
#: table used to turn a shape into a byte count.
_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
}
for _name, _code in (("float8_e4m3fn", "F8_E4M3"), ("float8_e4m3fnuz", "F8_E4M3FNUZ"),
                     ("float8_e5m2", "F8_E5M2"), ("float8_e5m2fnuz", "F8_E5M2FNUZ"),
                     ("uint16", "U16"), ("uint32", "U32"), ("uint64", "U64")):
    _dtype = getattr(torch, _name, None)
    if _dtype is not None:
        _DTYPES[_code] = _dtype
try:
    from safetensors.torch import _TYPES as _SAFETENSORS_TYPES
except ImportError:  # pragma: no cover - older or leaner builds keep it private
    _SAFETENSORS_TYPES = {}
for _code, _dtype in dict(_SAFETENSORS_TYPES).items():
    try:
        if torch.empty((), dtype=_dtype).element_size() > 0:
            _DTYPES.setdefault(str(_code).upper(), _dtype)
    except (RuntimeError, TypeError):  # pragma: no cover - a dtype this torch cannot instantiate
        continue

#: The container's own cap on the header, and the only sanity check worth making before trusting
#: the length prefix of a file that may not be a safetensors file at all.
_MAX_HEADER_BYTES = 100_000_000


def torch_dtype_of(code):
    dtype = _DTYPES.get(str(code).upper())
    if dtype is None:
        raise ValueError(f"safetensors dtype {code!r} is not one this build of torch can represent")
    return dtype


def _itemsize(dtype):
    return torch.empty((), dtype=dtype).element_size()


class ShardFormatError(ValueError):
    """This file is not a safetensors container this reader can read directly."""


@dataclasses.dataclass(frozen=True)
class TensorEntry:
    """One tensor's header entry: what it is, and where its bytes are in the file."""

    name: str
    dtype: object
    shape: tuple
    begin: int
    end: int

    @property
    def nbytes(self):
        return self.end - self.begin

    @property
    def rows(self):
        return self.shape[0] if self.shape else 0

    def row_span(self, start, stop):
        """``(file offset, byte count, shape)`` for rows ``[start, stop)`` of this tensor.

        Rows of a contiguous tensor are contiguous, so a run of them is one byte range and one
        read. This is what makes an expert cost its own bytes rather than its layer's, and it is
        the same arithmetic the mapped path's ``get_slice(key)[a:b]`` performs internally.
        """
        rows = self.rows
        if not rows:
            raise ValueError(f"{self.name} has no rows to slice")
        start, stop = int(start), int(stop)
        if not 0 <= start <= stop <= rows:
            raise IndexError(f"rows [{start}:{stop}) are outside {self.name}'s {rows}")
        row_bytes = self.nbytes // rows
        return (self.begin + start * row_bytes, (stop - start) * row_bytes,
                (stop - start,) + self.shape[1:])


class ShardIndex:
    """Every tensor in one shard, and where it lives. No tensor data, and no mapping."""

    __slots__ = ("path", "entries", "metadata", "size")

    def __init__(self, path, entries, metadata, size):
        self.path = str(path)
        self.entries = entries
        self.metadata = metadata
        self.size = size

    def keys(self):
        return list(self.entries)

    def __contains__(self, name):
        return name in self.entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, name):
        try:
            return self.entries[name]
        except KeyError:
            raise KeyError(f"{name!r} is not in {os.path.basename(self.path)}") from None

    def shapes(self):
        return {name: entry.shape for name, entry in self.entries.items()}


def parse_header(blob, size=None, path=""):
    """Turn the header of a safetensors file into ``(entries, metadata, data_start)``.

    Kept separate from the file so it can be tested on bytes, and so a malformed header is reported
    as one rather than as a mystery failure a hundred lines later.
    """
    what = path or "file"
    if len(blob) < 8:
        raise ShardFormatError(f"{what} is too short to be a safetensors container")
    (header_len,) = struct.unpack_from("<Q", blob, 0)
    if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
        raise ShardFormatError(f"{what} declares a {header_len}-byte header, which is not a length "
                               f"a safetensors container has")
    if len(blob) < 8 + header_len:
        raise ShardFormatError(f"{what} declares a {header_len}-byte header but only "
                               f"{len(blob) - 8} bytes follow the length")
    try:
        index = json.loads(bytes(blob[8:8 + header_len]))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ShardFormatError(f"{what} has a header that is not JSON: {exc}") from exc
    if not isinstance(index, dict):
        raise ShardFormatError(f"{what} has a header that is not an object")

    data_start = 8 + header_len
    metadata = index.pop("__metadata__", None)
    entries = {}
    for name, info in index.items():
        try:
            begin, end = info["data_offsets"]
            shape = tuple(int(d) for d in info["shape"])
            dtype = torch_dtype_of(info["dtype"])
        except (TypeError, KeyError, ValueError) as exc:
            raise ShardFormatError(f"{what} describes {name!r} in a way this reader does not "
                                   f"understand: {exc}") from exc
        begin, end = data_start + int(begin), data_start + int(end)
        count = 1
        for dim in shape:
            count *= dim
        if end - begin != count * _itemsize(dtype):
            raise ShardFormatError(f"{what} claims {end - begin} bytes for {name!r}, which is not "
                                   f"what its shape and dtype come to")
        if size is not None and end > size:
            raise ShardFormatError(f"{what} places {name!r} past the end of the file")
        entries[name] = TensorEntry(name=name, dtype=dtype, shape=shape, begin=begin, end=end)
    return entries, metadata, data_start


def read_index(path):
    """Parse one shard's header, reading only the header's own bytes off the disk."""
    path = str(path)
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) < 8:
            raise ShardFormatError(f"{path} is too short to be a safetensors container")
        (header_len,) = struct.unpack_from("<Q", prefix, 0)
        if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
            raise ShardFormatError(f"{path} declares a {header_len}-byte header, which is not a "
                                   f"length a safetensors container has")
        blob = prefix + handle.read(header_len)
    entries, metadata, _ = parse_header(blob, size=size, path=path)
    return ShardIndex(path, entries, metadata, size)


def close_handle(handle):
    """Unmap a shard handle now, rather than whenever the collector gets to it.

    safetensors exposes the release through the context-manager protocol and nothing else, so that
    is what is called.
    """
    closer = getattr(handle, "__exit__", None)
    if closer is None:
        return
    try:
        closer(None, None, None)
    except Exception:  # noqa: BLE001 - a handle that will not close must not fail a read
        log.debug("a shard handle did not close cleanly", exc_info=True)


def _read_range_into(handle, offset, nbytes, destination):
    """Read ``nbytes`` at ``offset`` into a 1-D uint8 tensor. One copy, file to destination."""
    if nbytes == 0:
        return
    handle.seek(offset)
    try:
        # torch tensors do not expose the buffer protocol; numpy shares the memory rather than
        # copying it, so the file writes straight into the tensor the caller will keep.
        view = destination.numpy()
    except (RuntimeError, ImportError):
        # No numpy for this tensor. Correct, one copy more expensive, and announced so a machine
        # that lands here is recognisable from its logs rather than only from its throughput.
        caps.announce_once("shard-read-buffered",
                           "reading shards through an intermediate buffer: this torch cannot hand "
                           "out a writable view of a tensor, so each read costs one extra copy",
                           logging.INFO)
        destination.copy_(torch.frombuffer(bytearray(handle.read(nbytes)), dtype=torch.uint8))
        return
    got = handle.readinto(view)
    if got != nbytes:
        raise EOFError(f"wanted {nbytes} bytes at offset {offset} of {handle.name!r} and the file "
                       f"gave {got}")


@dataclasses.dataclass(frozen=True)
class ReadRequest:
    """One tensor, or a run of its rows, and the buffer its bytes belong in."""

    name: str
    destination: object
    row_start: object = None
    row_stop: object = None

    @property
    def sliced(self):
        return self.row_start is not None


def direct_reads_available():
    """Whether shard bytes can be used exactly as they are stored.

    The container is little-endian by definition. Everywhere torch actually runs the host is too,
    and the bytes need no work at all; a big-endian host would need every element swapped, which
    safetensors already does correctly, so that machine takes the mapped path rather than growing a
    second implementation of byte swapping here.
    """
    return sys.byteorder == "little"


class ShardReader:
    """Reads tensors out of one checkpoint's shards, by the cheapest means the host allows.

    Instances are shared: every reader of a given checkpoint directory goes through the same one,
    so whatever bound applies applies to the process rather than to one call site. That sharing is
    the point. The engine reads shards from the thread running the forward, from the cache's
    prefetch workers and from the loader's io pool at the same time, and a bound each of them
    applied separately would be three bounds and no ceiling.
    """

    DIRECT = "direct"
    MAPPED = "mapped"

    #: Handle-limit settings other than a plain count.
    AUTO = "auto"
    UNBOUNDED = "unbounded"

    def __init__(self, root, profile=None, shard_handle_limit=AUTO, readers=1):
        self.root = str(root)
        self.readers = max(1, int(readers or 1))
        self._indexes = {}
        self._indexes_lock = threading.Lock()
        #: Shards the header reader declined, which are read through a mapping instead. Per shard
        #: rather than per checkpoint: one unreadable file must not cost the rest their fast path.
        self._mapped_paths = set()
        self.reads = 0
        self.bytes_read = 0
        self.handle_opens = 0
        self.handle_evictions = 0
        self._local = threading.local()
        self._handles = set()
        self._handles_lock = threading.Lock()

        self.mode, self.handle_limit, self.reason = self._resolve_strategy(
            profile, shard_handle_limit)
        self.handle_mode = "bounded" if self.handle_limit > 0 else "unbounded"
        #: Handles per reader thread. Zero means unbounded; otherwise the total is shared out, so
        #: the bound holds across the pool rather than being applied to each worker separately.
        self._per_thread_limit = (0 if self.handle_limit <= 0
                                  else max(1, self.handle_limit // self.readers))
        if self._per_thread_limit:
            # A reader cannot hold fewer than the one handle it is reading through, so a limit
            # below the reader count cannot be met. Publish what will actually be held rather than
            # what was asked for: a bound that is quietly exceeded is worse than an honest one.
            # The other direction needs no adjustment -- a limit that does not divide evenly leaves
            # the pool holding fewer than it was allowed, and the number asked for is still a true
            # ceiling.
            effective = self._per_thread_limit * self.readers
            if effective > self.handle_limit:
                self.reason += (f" (raised from {self.handle_limit} to {effective}: "
                                f"{self.readers} readers cannot share fewer)")
                self.handle_limit = effective
        if self.mode == self.DIRECT:
            caps.announce_once(
                "shard-read-direct",
                f"reading shards by byte range rather than mapping them: {self.reason}. Each read "
                f"costs one copy into the engine's own buffer and nothing is held mapped.",
                logging.INFO)
        elif self.handle_mode == "bounded":
            caps.announce_once(
                "shard-handle-bound",
                f"holding at most {self.handle_limit} shard file(s) mapped at a time "
                f"({self._per_thread_limit} per reader): {self.reason}. Reads are unaffected; a "
                f"shard is reopened when it comes round again.", logging.INFO)

    @property
    def handle_reason(self):
        """Kept because ``reason`` reads better and this name is already in the field reports."""
        return self.reason

    # -- what the machine will let us hold mapped ------------------------------------------------

    def checkpoint_bytes(self):
        """What this checkpoint would cost to hold mapped in its entirety."""
        try:
            return sum(entry.stat().st_size for entry in os.scandir(self.root)
                       if entry.name.endswith(".safetensors"))
        except OSError:
            return 0

    def largest_shard_bytes(self):
        try:
            return max((entry.stat().st_size for entry in os.scandir(self.root)
                        if entry.name.endswith(".safetensors")), default=0)
        except OSError:
            return 0

    def _resolve_strategy(self, profile, setting):
        """``(mode, handle limit, why)``. A limit of 0 means nothing bounds what stays mapped.

        Mapping a shard is the cheaper read when the machine can afford it. safetensors hands back
        tensors that ALIAS the mapping rather than copies of it -- measured: every tensor in a
        1.57GB shard came back pointing into one 1.57GB region, in four milliseconds -- so a read
        that maps costs no copy at all, and the page cache serves a second read of the same shard
        for free.

        The catch is the same aliasing. A tensor that aliases the mapping keeps the mapping alive
        for as long as the tensor lives, whatever the handle does, so one cached 4MB expert holds
        its whole 1.57GB shard. Where mappings are charged against a commit limit that is fatal,
        and it is why bounding the handles alone did not fix this: closing a handle whose tensors
        are still held frees nothing.

        So the choice is a comparison of two measured numbers -- what this checkpoint would cost
        held mapped, against what this machine will let the process charge -- and never a platform
        test. A machine that does not charge for mappings publishes no budget and maps freely
        without this knowing or caring which machine it is.
        """
        if isinstance(setting, str):
            if setting == self.UNBOUNDED:
                return self.MAPPED, 0, "requested explicitly"
            if setting == self.DIRECT:
                return self._direct_or_bounded(profile, "requested explicitly")
            if setting != self.AUTO:
                raise ValueError(
                    f"shard_handle_limit must be {self.AUTO!r}, {self.UNBOUNDED!r}, "
                    f"{self.DIRECT!r} or an integer, not {setting!r}")
        elif setting is not None:
            limit = int(setting)
            if limit <= 0:
                return self._direct_or_bounded(profile, "requested explicitly (no shard mapped)")
            return self.MAPPED, limit, f"requested explicitly ({limit})"

        budget = self._mapping_budget(profile)[0]
        if budget <= 0:
            return self.MAPPED, 0, ("this machine does not charge memory mappings against a commit "
                                    "limit, or it could not be measured")

        footprint = self.checkpoint_bytes()
        if footprint <= budget:
            # Includes a footprint of zero, which means the shards could not be measured or are not
            # there yet. Giving up the cheaper read on the strength of a number we do not have
            # would trade a real cost for a risk there is no evidence of.
            return self.MAPPED, 0, (f"the whole checkpoint ({footprint / 1024 ** 3:.1f}GB) fits in "
                                    f"the {budget / 1024 ** 3:.1f}GB that may be held mapped")

        return self._direct_or_bounded(profile, (
            f"the checkpoint would cost {footprint / 1024 ** 3:.1f}GB held mapped against a "
            f"{budget / 1024 ** 3:.1f}GB budget"))

    def _direct_or_bounded(self, profile, why):
        """Read by byte range, or -- where that is impossible -- map as little as will serve.

        The fallback is for a big-endian host, which cannot use the container's little-endian
        payload as stored. It has to map, so all that is left is to map as few shards at a time as
        the reads allow, which is what the floor and ceiling are for.
        """
        if direct_reads_available():
            return self.DIRECT, 0, f"{why}, so shards are read by byte range instead"
        budget, floor, ceiling = self._mapping_budget(profile)
        largest = self.largest_shard_bytes()
        affordable = int(budget // largest) if largest and budget else ceiling
        limit = max(floor, min(ceiling, affordable))
        return self.MAPPED, limit, (
            f"{why}, and this host is big-endian so the payload cannot be read as stored; the "
            f"largest shard is {largest / 1024 ** 3:.2f}GB")

    @staticmethod
    def _mapping_budget(profile):
        """``(bytes that may be held mapped, floor, ceiling)``.

        The share is the profile's; the headroom it is a share OF is measured here and now. A
        profile is probed once and cached on disk, and commit headroom is the one number in it that
        moves by gigabytes between one run and the next -- another process starting, or this one
        having already built a model. Deciding from a cached figure would map a checkpoint against
        headroom the machine had yesterday, which is the failure this whole path exists to avoid.
        A manual override is left exactly as it was set.
        """
        budget, floor, ceiling = 0, 2, 4
        fraction = None
        overridden = False
        if profile is not None:
            derivation = profile.derived.get("shard_mapping_budget_bytes")
            if derivation is not None:
                budget = max(0, int(derivation.value))
                fraction = (derivation.inputs or {}).get("shard_mapping_fraction")
                overridden = getattr(derivation, "source", None) == "override"
            inputs = getattr(profile.derived.get("shard_handle_limit"), "inputs", None) or {}
            floor = int(inputs.get("shard_handle_floor", floor))
            ceiling = int(inputs.get("shard_handle_ceiling", ceiling))
        if budget > 0 and fraction and not overridden:
            live = caps.commit_headroom()
            if live.measured and live.charges_mappings:
                budget = int(live.available * float(fraction))
        return budget, floor, ceiling

    # -- headers -----------------------------------------------------------------------------

    def index(self, path):
        """This shard's header, parsed once and remembered.

        Keyed on size and modification time as well as path, so a checkpoint rewritten under a
        running process is re-read rather than served from a stale index. Holding one costs a few
        kilobytes and no mapping, which is the whole reason the engine can now size a 67GB model
        without charging a byte for it.
        """
        path = str(path)
        try:
            stat = os.stat(path)
            stamp = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            stamp = None
        with self._indexes_lock:
            cached = self._indexes.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        try:
            index = read_index(path)
        except ShardFormatError as exc:
            # Something in this file is beyond what the header reader understands -- most likely a
            # dtype safetensors gained and torch cannot size, which is where a sub-byte format
            # would land. safetensors itself can still read it, so this shard goes back to being
            # mapped rather than the model refusing to load over one tensor.
            caps.announce_once(
                "shard-read-fallback",
                f"reading some shards through memory mappings: {exc}", logging.INFO)
            log.debug("falling back to a mapped read of %s", path, exc_info=True)
            index = self._mapped_index(path)
        with self._indexes_lock:
            self._indexes[path] = (stamp, index)
        return index

    def _mapped_index(self, path):
        """A shard's index built by safetensors, for a file the header reader declined.

        Byte offsets are left at zero because nothing reads them: a shard indexed this way is read
        through its mapping, where the offsets are safetensors' business. What is needed here is
        the dtype and shape, which is all the rest of the engine asks an index for.
        """
        from safetensors import safe_open

        entries = {}
        with safe_open(path, framework="pt") as handle:
            self._mapped_paths.add(path)
            for name in handle.keys():
                entry = handle.get_slice(name)
                shape = tuple(int(d) for d in entry.get_shape())
                try:
                    dtype = torch_dtype_of(entry.get_dtype())
                except ValueError:
                    # The code is unknown to this reader but not to safetensors, so ask for an
                    # empty slice and take the dtype off the tensor it hands back. Reads no data.
                    dtype = (entry[0:0] if shape and shape[0] else handle.get_tensor(name)).dtype
                count = 1
                for dim in shape:
                    count *= dim
                nbytes = count * _itemsize(dtype)
                entries[name] = TensorEntry(name=name, dtype=dtype, shape=shape, begin=0,
                                            end=nbytes)
        return ShardIndex(path, entries, None, os.path.getsize(path))

    def keys(self, path):
        return self.index(path).keys()

    def shapes(self, path):
        return self.index(path).shapes()

    # -- the mapped fallback -------------------------------------------------------------------

    def _mapped_handle(self, path):
        """A mapped handle for this shard, opened if the calling thread does not hold one.

        Least-recently-used, and eviction closes the handle rather than dropping the reference: an
        unmapped section is what gives the commit charge back, and waiting for the collector to
        notice defeats the point of bounding this at all.

        Per thread, never shared, so eviction needs no coordination -- a handle can only be evicted
        by the thread that owns it, which is not inside a read when it does so, and every read
        copies out of the mapping before returning.
        """
        from safetensors import safe_open

        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = self._local.handles = collections.OrderedDict()
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
            close_handle(evicted)
        return handle

    def _mapped_tensors(self, path, index, names, rows):
        """Tensors straight out of the mapping, with no copy at all.

        safetensors aliases rather than copies, so this hands back the file's own pages. It is the
        cheapest read there is, and it is only reachable once the mode decision has established
        that this machine can afford to have those pages held for as long as the caller keeps them.
        """
        handle = self._mapped_handle(str(path))
        out = {}
        for name in names:
            entry = index[name]
            wanted = rows.get(name)
            if wanted is None:
                out[name] = handle.get_tensor(name)
                self.reads += 1
                self.bytes_read += entry.nbytes
                continue
            source = handle.get_slice(name)
            parts = [source[int(row):int(row) + 1] for row in wanted]
            out[name] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            self.reads += len(parts)
            self.bytes_read += (entry.nbytes // entry.rows if entry.rows else 0) * len(parts)
        return out

    def _read_mapped(self, path, requests, index):
        handle = self._mapped_handle(path)
        for request in requests:
            entry = index[request.name]
            if request.sliced:
                source = handle.get_slice(request.name)[request.row_start:request.row_stop]
                shape = entry.row_span(request.row_start, request.row_stop)[2]
            else:
                # get_tensor rather than a full slice: slicing is not defined for a 0-dim tensor,
                # and a checkpoint that stores a scalar is not a checkpoint this may refuse.
                source = handle.get_tensor(request.name)
                shape = entry.shape
            request.destination.view(entry.dtype).view(shape).copy_(source)
            self.reads += 1
            self.bytes_read += request.destination.numel()

    # -- reading -----------------------------------------------------------------------------

    def read_into(self, path, requests):
        """Fill each request's buffer with that tensor's bytes, or those of a run of its rows.

        Requests are served in file order rather than in the order they arrive. The destinations
        are disjoint buffers, so the order changes nothing about the result, and reading a shard
        front to back is the access pattern every storage tier is fastest at -- most of all the
        rotational one, where seeking between two tensors can cost more than reading either.
        """
        path = str(path)
        requests = list(requests)
        if not requests:
            return
        index = self.index(path)
        if self.mode == self.MAPPED or path in self._mapped_paths:
            self._read_mapped(path, requests, index)
            return

        plan = []
        for request in requests:
            entry = index[request.name]
            if request.sliced:
                offset, nbytes, _ = entry.row_span(request.row_start, request.row_stop)
            else:
                offset, nbytes = entry.begin, entry.nbytes
            if nbytes != request.destination.numel():
                raise ValueError(f"{request.name} needs {nbytes} bytes and was given a "
                                 f"{request.destination.numel()}-byte buffer")
            plan.append((offset, nbytes, request.destination))
        plan.sort(key=lambda item: item[0])

        with open(path, "rb") as handle:
            for offset, nbytes, destination in plan:
                _read_range_into(handle, offset, nbytes, destination)
                self.reads += 1
                self.bytes_read += nbytes

    def partition(self, path, requests, ways):
        """Split reads into at most ``ways`` batches, each one a contiguous sweep of the file.

        Handing every reader an arbitrary subset would turn one sequential pass over a shard into
        as many interleaved seek storms as there are workers, which costs nothing on an NVMe and
        costs a rotational device most of its bandwidth. Cutting the file into adjacent stretches
        instead keeps each worker sequential within its own, so widening the pool overlaps whole
        regions rather than individual tensors.

        Batches are balanced by bytes, not by count: a layer is one 500MB projection and thirty
        small norms, and splitting it evenly by count would leave one worker doing all the work.
        """
        index = self.index(str(path))
        sized = []
        for request in requests:
            entry = index[request.name]
            if request.sliced:
                offset, nbytes, _ = entry.row_span(request.row_start, request.row_stop)
            else:
                offset, nbytes = entry.begin, entry.nbytes
            sized.append((offset, nbytes, request))
        if not sized:
            return []
        sized.sort(key=lambda item: item[0])

        ways = max(1, min(int(ways), len(sized)))
        total = sum(nbytes for _, nbytes, _ in sized)
        bounds = [total * (i + 1) / ways for i in range(ways)]
        batches = [[] for _ in range(ways)]
        seen = 0
        at = 0
        for _, nbytes, request in sized:
            batches[at].append(request)
            seen += nbytes
            while at < ways - 1 and seen >= bounds[at]:
                at += 1
        return [batch for batch in batches if batch]

    def read_tensors(self, path, keys=None, rows=None):
        """``{name: tensor}`` for the requested keys, reading only their bytes.

        ``rows`` maps a name to the row indices wanted out of it, for the fused expert layout where
        one row is one expert. Those come back compacted into one tensor, rows in the order asked
        for, which is what the caller scatters into the full-width parameter on the device.
        """
        index = self.index(path)
        names = list(index.keys() if keys is None else keys)
        rows = rows or {}
        if self.mode == self.MAPPED or str(path) in self._mapped_paths:
            return self._mapped_tensors(path, index, names, rows)

        out = {}
        requests = []
        for name in names:
            entry = index[name]
            wanted = rows.get(name)
            if wanted is None:
                buffer = torch.empty(entry.nbytes, dtype=torch.uint8)
                out[name] = buffer.view(entry.dtype).view(entry.shape)
                requests.append(ReadRequest(name=name, destination=buffer))
                continue
            wanted = [int(row) for row in wanted]
            row_bytes = entry.nbytes // entry.rows if entry.rows else 0
            buffer = torch.empty(row_bytes * len(wanted), dtype=torch.uint8)
            out[name] = buffer.view(entry.dtype).view((len(wanted),) + entry.shape[1:])
            # One request per run of consecutive rows. A mixture's top-k usually is not
            # consecutive, but where it is this reads the run in one go instead of row by row.
            start = 0
            while start < len(wanted):
                stop = start + 1
                while stop < len(wanted) and wanted[stop] == wanted[stop - 1] + 1:
                    stop += 1
                requests.append(ReadRequest(
                    name=name,
                    destination=buffer.narrow(0, start * row_bytes, (stop - start) * row_bytes),
                    row_start=wanted[start], row_stop=wanted[stop - 1] + 1))
                start = stop
        self.read_into(path, requests)
        return out

    # -- lifecycle ---------------------------------------------------------------------------

    def stats(self):
        return {
            "shard_read_mode": self.mode,
            "shard_read_reason": self.reason,
            "shard_reads": self.reads,
            "shard_bytes_read": self.bytes_read,
            # Reported because a bounded run reopens shards, and "why is this slower than the same
            # model on the other box" is answered by these numbers and nothing else.
            "shard_handle_mode": self.handle_mode,
            "shard_handle_limit": self.handle_limit,
            "shard_handle_opens": self.handle_opens,
            "shard_handle_evictions": self.handle_evictions,
        }

    def release(self):
        """Drop every mapped handle and every parsed header.

        Called when a model closes. On Windows a mapped shard cannot be deleted, so leaving it to
        the collector would keep the checkpoint undeletable for an unbounded time.
        """
        with self._handles_lock:
            handles, self._handles = self._handles, set()
        for handle in handles:
            close_handle(handle)
        self._local = threading.local()
        with self._indexes_lock:
            self._indexes.clear()
            self._mapped_paths.clear()


# -- one reader per checkpoint ------------------------------------------------------------------

_readers = {}
_readers_lock = threading.Lock()


def _key(root):
    return os.path.normcase(os.path.abspath(str(root)))


def reader_for(root, profile=None, shard_handle_limit=None, readers=None):
    """The shared reader for a checkpoint directory, created on first use.

    Sharing is what makes any bound meaningful, and it is also what lets the free functions in
    ``rocketllm.utils`` -- which are handed a path and nothing else -- read through the settings the
    engine resolved when it opened the model.
    """
    key = _key(root)
    with _readers_lock:
        reader = _readers.get(key)
        if reader is not None:
            return reader
    # Built outside the lock: probing a checkpoint's shard sizes touches the filesystem, and
    # holding a global lock across that would serialise every reader in the process behind it.
    built = ShardReader(root, profile=profile, readers=readers or 1,
                        shard_handle_limit=(ShardReader.AUTO if shard_handle_limit is None
                                            else shard_handle_limit))
    with _readers_lock:
        return _readers.setdefault(key, built)


def configure_reader(root, profile=None, shard_handle_limit=None, readers=None):
    """Build this checkpoint's reader with the engine's settings, replacing any default one.

    The engine calls this when it opens a model. A reader created earlier by a free function knows
    nothing about the hardware profile or the user's overrides, and those are exactly the settings
    that decide how this behaves.
    """
    key = _key(root)
    reader = ShardReader(root, profile=profile, readers=readers or 1,
                         shard_handle_limit=(ShardReader.AUTO if shard_handle_limit is None
                                             else shard_handle_limit))
    with _readers_lock:
        previous = _readers.get(key)
        _readers[key] = reader
    if previous is not None:
        previous.release()
    return reader


def release_reader(root, only=None):
    """Drop a checkpoint's reader and everything it holds open.

    ``only`` makes it a no-op unless the registered reader is that one, so a model shutting down
    cannot pull the reader out from under a second model that opened the same checkpoint after it.
    """
    with _readers_lock:
        reader = _readers.get(_key(root))
        if reader is None or (only is not None and reader is not only):
            return None
        del _readers[_key(root)]
    reader.release()
    return reader


def release_all():
    """Drop every reader. For tests, and for anything that has to leave the files alone."""
    with _readers_lock:
        readers = list(_readers.values())
        _readers.clear()
    for reader in readers:
        reader.release()
