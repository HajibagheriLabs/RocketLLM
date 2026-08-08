"""Host staging buffers for the host -> device path.

A layer is packed into one contiguous host buffer and sent in a single transfer. The buffer wants
to be page-locked, because pageable memory forces the driver to stage the copy through its own
bounce buffer -- but page-locking is a synchronizing driver call, measured on the machine this was
written on at 15.1ms for a layer-sized buffer against 0.03ms for an ordinary allocation. Paid once
per layer per token, that costs more than the transfer it is supposed to accelerate.

So buffers are allocated once per size class and handed back out. Layers of a given architecture
are near enough the same size that a handful of classes covers a whole model.
"""
import logging
import threading

import torch

from ..hw import caps

log = logging.getLogger(__name__)

#: Smallest buffer worth pooling. Below this, allocation is cheap enough that pooling only adds
#: bookkeeping, and rounding up wastes a larger share of what it hands out.
_MIN_POOLED_ELEMENTS = 1 << 16


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
        self._resident_bytes = 0
        self._lock = threading.Lock()
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

    def stats(self):
        with self._lock:
            return {
                "buffers": len(self._buffers),
                "resident_bytes": self._resident_bytes,
                "budget_bytes": self.budget_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "pinned": self.caps.can_pin_memory,
            }

    def clear(self):
        """Drop every pooled buffer. Between generations, not between layers."""
        with self._lock:
            self._buffers.clear()
            self._resident_bytes = 0
