"""Host staging buffers for the host -> device path.

A layer is packed into one contiguous host buffer and sent in a single transfer. The buffer wants
to be page-locked, because pageable memory forces the driver to stage the copy through its own
bounce buffer -- but page-locking is a synchronizing driver call, measured on the machine this was
written on at 15.1ms for a layer-sized buffer against 0.03ms for an ordinary allocation. Paid once
per layer per token, that costs more than the transfer it is supposed to accelerate.

So buffers are allocated once per size class and handed back out. Layers of a given architecture
are near enough the same size that a handful of classes covers a whole model.

Buffers are LEASED, not merely borrowed, and that distinction is a correctness requirement rather
than bookkeeping. Once a transfer is issued asynchronously the DMA engine keeps reading the host
buffer after the call returns, so handing the same buffer to the next layer's loader would have it
overwritten mid-flight. The result is not a crash: it is a few wrong bytes in a weight, which
produces plausible-looking wrong tokens and nothing else. A lease is only released once its
transfer's event has actually fired -- see rocketllm/streaming/transfer.py.
"""
import logging
import threading

import torch

from ..hw import caps

log = logging.getLogger(__name__)

#: Smallest buffer worth pooling. Below this, allocation is cheap enough that pooling only adds
#: bookkeeping, and rounding up wastes a larger share of what it hands out.
_MIN_POOLED_ELEMENTS = 1 << 16


class BufferLease:
    """A host buffer checked out of the pool, and the only thing allowed to give it back.

    Holding one is a promise that nothing else is writing to those bytes. Releasing it while a
    transfer is still reading from it is the corruption this whole mechanism exists to prevent, so
    release goes through the transfer's completion guard rather than being called directly.
    """

    __slots__ = ("view", "_pool", "_slot", "released")

    def __init__(self, view, pool=None, slot=None):
        self.view = view
        self._pool = pool
        self._slot = slot
        self.released = False

    @property
    def pooled(self):
        return self._slot is not None

    def release(self):
        if self.released:
            return
        self.released = True
        if self._pool is not None and self._slot is not None:
            self._pool._return(self._slot)
        self.view = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


class _Slot:
    """One pooled buffer and whether it is currently checked out."""

    __slots__ = ("tensor", "dtype", "leased")

    def __init__(self, tensor, dtype):
        self.tensor = tensor
        self.dtype = dtype
        self.leased = False


class HostStagingPool:
    """Reusable host buffers, bucketed by size class and dtype.

    Not thread-safe by accident -- it takes a lock, because expert streaming and layer streaming
    both come through here and a future reader thread would too.

    A budget of zero is a supported configuration, not a failure: it means no pooling and no
    pinning, every buffer transient and pageable. That is the degradation path for a machine with
    no free RAM to spare, and it still produces correct output.
    """

    def __init__(self, device_caps, budget_bytes):
        self.caps = device_caps
        self.budget_bytes = max(0, int(budget_bytes or 0))
        self._buffers = {}
        #: Leasable buffers, for the asynchronous path. Kept separate from `_buffers` because the
        #: two have different lifetimes: a `buffer()` result is finished with by the time the call
        #: that asked for it returns, a lease is not.
        self._slots = []
        self._reclaim = None
        self._resident_bytes = 0
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

        if self.budget_bytes == 0:
            caps.announce_once(
                "staging-pool-disabled",
                "the host staging pool budget computed to 0, so staging buffers are allocated per "
                "layer and never page-locked. Correct, but transfers will be slower.",
                logging.INFO)

    @staticmethod
    def _size_class(count):
        """Round up to the next power of two, so near-identical layers share one buffer."""
        if count <= _MIN_POOLED_ELEMENTS:
            return _MIN_POOLED_ELEMENTS
        return 1 << (count - 1).bit_length()

    def buffer(self, count, dtype):
        """A host buffer of at least `count` elements, as a view of exactly that many.

        The returned view is only valid until the next call for the same size class, which is
        fine because a layer is packed and sent before the next one is staged.
        """
        if count <= 0:
            return torch.empty(0, dtype=dtype)

        if self.budget_bytes == 0:
            self.misses += 1
            return torch.empty(count, dtype=dtype)

        size_class = self._size_class(count)
        itemsize = torch.empty((), dtype=dtype).element_size()

        with self._lock:
            # Any buffer of this dtype that is already big enough will do; take the smallest, so a
            # large one held for the embedding is not handed out for every little norm. Matching
            # only the exact size class would miss it and allocate a second buffer for two layers
            # that happen to straddle a power of two.
            fitting = [(key, buffer) for key, buffer in self._buffers.items()
                       if key[1] == dtype and buffer.numel() >= count]
            if fitting:
                _, buffer = min(fitting, key=lambda pair: pair[1].numel())
                self.hits += 1
                return buffer.narrow(0, 0, count)

            nbytes = size_class * itemsize
            # A new buffer for this dtype supersedes every smaller one: they can only serve
            # requests it can serve too. Releasing them keeps the pool converging on one buffer
            # per dtype, sized to the largest layer, which is the smallest thing that can stage
            # this model at all.
            superseded = [key for key, buffer in self._buffers.items()
                          if key[1] == dtype and buffer.numel() < size_class]
            reclaimed = sum(self._buffers[key].numel() * itemsize for key in superseded)

            if self._resident_bytes - reclaimed + nbytes > self.budget_bytes:
                # Over budget. Rather than evict something an in-flight transfer might still be
                # reading from, hand back a transient buffer: slower, always correct.
                caps.announce_once(
                    "staging-pool-full",
                    f"the host staging pool reached its budget of "
                    f"{self.budget_bytes / 1024 ** 2:.0f}MB; larger layers fall back to transient "
                    f"buffers, which still work but are allocated every time.",
                    logging.INFO)
                self.misses += 1
                return torch.empty(count, dtype=dtype)

            for key in superseded:
                del self._buffers[key]
            self._resident_bytes -= reclaimed

            buffer = self.caps.pinned_empty((size_class,), dtype)
            self._buffers[(size_class, dtype)] = buffer
            self._resident_bytes += nbytes
            self.misses += 1
            return buffer.narrow(0, 0, count)

    # -- leases ---------------------------------------------------------------------------------

    def lease(self, count, dtype=torch.uint8):
        """Check out a buffer of at least `count` elements that nobody else can be handed.

        Unlike :meth:`buffer`, which is fine for a copy that completes before it returns, a lease
        survives an asynchronous transfer. The pool will grow to as many concurrent leases as the
        budget allows -- which is what lets the next layer be staged while the previous one is still
        crossing the link -- and past that point it hands out an unpooled buffer rather than
        recycling one that is still in flight. Slower, never wrong.
        """
        if count <= 0:
            return BufferLease(torch.empty(0, dtype=dtype))

        itemsize = torch.empty((), dtype=dtype).element_size()
        with self._lock:
            slot = self._free_slot(count, dtype)
            if slot is None and self._reclaim is not None:
                # Ask the transfer layer whether anything has finished; a completed transfer's
                # buffer is free even though nobody has got round to saying so.
                self._lock.release()
                try:
                    self._reclaim()
                finally:
                    self._lock.acquire()
                slot = self._free_slot(count, dtype)

            if slot is None:
                size_class = self._size_class(count)
                nbytes = size_class * itemsize
                if self.budget_bytes and self._resident_bytes + nbytes <= self.budget_bytes:
                    slot = _Slot(self.caps.pinned_empty((size_class,), dtype), dtype)
                    self._slots.append(slot)
                    self._resident_bytes += nbytes
                    self.misses += 1

            if slot is not None:
                slot.leased = True
                self.hits += 1 if slot.tensor.numel() >= count else 0
                return BufferLease(slot.tensor.narrow(0, 0, count), self, slot)

        # Every pooled buffer is still in flight, or there is no budget at all. A transient buffer
        # is always safe: it belongs to this lease alone and is freed when the lease is.
        caps.announce_once(
            "staging-transient",
            "the host staging pool had no free buffer for a transfer already in flight, so this "
            "one uses a transient pageable buffer. Correct, but allocated every time; a larger "
            "staging budget would avoid it.", logging.INFO)
        self.misses += 1
        return BufferLease(torch.empty(count, dtype=dtype))

    def _free_slot(self, count, dtype):
        """The smallest un-leased buffer that can serve `count`, so big ones stay for big layers."""
        fitting = [s for s in self._slots
                   if not s.leased and s.dtype == dtype and s.tensor.numel() >= count]
        return min(fitting, key=lambda s: s.tensor.numel()) if fitting else None

    def _return(self, slot):
        with self._lock:
            slot.leased = False

    def set_reclaim_hook(self, hook):
        """Let the pool ask the transfer layer to harvest finished transfers before it allocates."""
        self._reclaim = hook

    @property
    def leased(self):
        with self._lock:
            return sum(1 for s in self._slots if s.leased)

    def stats(self):
        with self._lock:
            return {
                "buffers": len(self._buffers),
                "slots": len(self._slots),
                "leased": sum(1 for s in self._slots if s.leased),
                "resident_bytes": self._resident_bytes,
                "budget_bytes": self.budget_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "pinned": self.caps.can_pin_memory,
            }

    def clear(self):
        """Drop every pooled buffer. Between generations, not between layers.

        A leased slot is deliberately kept: something is still reading from it, and freeing it here
        would be the same corruption releasing it early would cause.
        """
        with self._lock:
            self._buffers.clear()
            self._slots = [s for s in self._slots if s.leased]
            # Recomputed rather than decremented: everything that is not still checked out has
            # gone, so what remains resident is exactly what is still in flight.
            self._resident_bytes = sum(s.tensor.numel() * s.tensor.element_size()
                                       for s in self._slots)
