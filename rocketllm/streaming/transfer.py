"""Getting a staged layer across the link without blocking the thread that asked for it.

The transfer is issued on a dedicated copy stream, one non-blocking copy for the whole staged
buffer, and an event is recorded behind it. Whoever needs the weights makes the *compute* stream
wait on that event -- a dependency between two streams, resolved by the device -- rather than making
the CPU wait. That is the difference between a transfer that costs wall-clock time and one that
hides behind the compute of the layer before it.

Backends without streams take a synchronous path chosen by capability query, never by backend name,
and produce identical results more slowly.

The hazard here, and the reason this module is more careful than its size suggests: after an
asynchronous copy is issued the DMA engine goes on reading the host buffer, and the call has already
returned. Give that buffer back to the pool and the next layer's loader writes over bytes still in
flight. Nothing raises. A few weights come out wrong, the model keeps generating, and the output is
subtly bad in a way that looks like a bad checkpoint. So a staged buffer's lease is released in
exactly one place -- :meth:`TransferHandle._release_source` -- which refuses to release until the
event has actually fired, and both the async and the sync path go through it.
"""
import logging
import threading

import torch

from ..hw import caps

log = logging.getLogger(__name__)


class TransferHandle:
    """One in-flight host -> device copy and the host buffer it must outlive.

    Returned by :meth:`WeightTransfer.send` before the copy has necessarily completed. Call
    :meth:`resolve` to get the device buffer with the compute stream correctly ordered behind it.
    """

    __slots__ = ("device_buffer", "event", "_lease", "_transfer", "_resolved", "nbytes", "_lock")

    def __init__(self, device_buffer, event, lease, transfer):
        self.device_buffer = device_buffer
        self.event = event
        self._lease = lease
        self._transfer = transfer
        self._resolved = False
        self._lock = threading.Lock()
        self.nbytes = device_buffer.numel() * device_buffer.element_size()

    # -- completion ----------------------------------------------------------------------------

    def is_complete(self):
        """Whether the copy has finished, without blocking to find out."""
        try:
            return bool(self.event.query())
        except Exception:  # noqa: BLE001 - an unqueryable event is treated as still running
            return False

    def synchronize(self):
        """Block this thread until the copy has finished."""
        self.event.synchronize()

    def _release_source(self):
        """Give the staged host buffer back -- and only once it is provably safe to.

        THE GUARD. Every release of a staging lease goes through here. If the event has not fired
        the copy may still be reading those bytes, so this blocks rather than handing them back:
        a stall is a performance problem, releasing early is a correctness one, and only one of
        those is recoverable.
        """
        with self._lock:
            lease = self._lease
            if lease is None:
                return False
            if not self.is_complete():
                self.synchronize()
            self._lease = None
        lease.release()
        return True

    def reclaim_if_complete(self):
        """Release the staged buffer if the copy has finished. Never blocks."""
        if self._lease is None:
            return True
        if not self.is_complete():
            return False
        return self._release_source()

    # -- what callers want ---------------------------------------------------------------------

    def resolve(self):
        """The device buffer, with the compute stream ordered behind the copy.

        The wait is queued on the device, so this returns immediately and the compute that follows
        simply will not start until the bytes have landed. This is the call the cache makes inside
        acquire, and it is where the overlap actually comes from.
        """
        if not self._resolved:
            self._transfer._await(self)
            self._resolved = True
        return self.device_buffer

    def __repr__(self):
        return (f"<TransferHandle {self.nbytes}B "
                f"{'complete' if self.is_complete() else 'in flight'}"
                f"{' holding a lease' if self._lease is not None else ''}>")


class SyncTransferHandle(TransferHandle):
    """The synchronous path's handle.

    The copy has already finished by the time this exists, so resolving is a no-op and the lease is
    safe to release at once. It still goes through the same guard, so there is one release path in
    this module and not two -- the second one is always the one that gets it wrong.
    """

    __slots__ = ()

    def is_complete(self):
        return True

    def synchronize(self):
        pass


class WeightTransfer:
    """The host -> device path: one copy stream, one transfer per staged layer.

    Not thread-safe by accident. The loader may stage on a worker thread while the main thread
    resolves, so the in-flight list takes a lock.
    """

    def __init__(self, device_caps, pool=None, device=None):
        self.caps = device_caps if device_caps is not None else caps.get_caps(device,
                                                                              announce=False)
        self.device = self.caps.device
        self.pool = pool
        self.stream = self.caps.copy_stream()
        self.is_async = bool(getattr(self.stream, "is_async", False))
        self._in_flight = []
        self._lock = threading.Lock()
        self.transfers = 0
        self.bytes_sent = 0
        self.stalls = 0

        if pool is not None and hasattr(pool, "set_reclaim_hook"):
            # Let the pool harvest finished transfers before it decides it has no free buffer.
            pool.set_reclaim_hook(self.reclaim)

        if not self.is_async:
            caps.announce_once(
                f"transfer-sync-{self.caps.backend}",
                f"the {self.caps.backend} backend has no copy streams, so weight transfers run on "
                f"the synchronous path and do not overlap with compute. Correct, and slower by "
                f"roughly the transfer time of every layer.", logging.INFO)

    # -- issuing -------------------------------------------------------------------------------

    def send(self, staged):
        """Send a staged layer's whole buffer in one transfer.

        `staged` is a LoadedLayer: it carries the host buffer and, crucially, the lease that must
        not be released until this transfer has finished with it.
        """
        return self.send_buffer(staged.buffer, staged.lease)

    def send_buffer(self, host, lease):
        """Send an already-staged host buffer, taking ownership of its lease.

        Separate from :meth:`send` because the engine packs some layers itself, by dtype, rather
        than through the loader's byte layout. Either way the lease crosses over here and is not
        the caller's to release afterwards.
        """
        if self.is_async:
            return self._send_async(host, lease)
        return self._send_sync(host, lease)

    @staticmethod
    def _unaliased(device_buffer, host):
        """Guarantee the device buffer is its own memory rather than a view of the staged one.

        ``Tensor.to(device)`` returns *self* when the tensor is already on that device. So on a
        backend whose device is the host -- the CPU path, and any build where the copy is elided
        because the two share memory -- the "device" buffer IS the staging buffer. Parameters are
        then bound as views into it and the lease goes back to the pool, and the next staged read
        writes straight over live weights.

        This is the same hazard the module docstring describes, one tier down, and it fails the same
        way: nothing raises, a few weights come out wrong, and the output is subtly bad. Copying
        costs nothing on a real accelerator, where the transfer already produced distinct memory and
        this check simply never fires.
        """
        if device_buffer.data_ptr() == host.data_ptr():
            return host.clone()
        return device_buffer

    def _send_async(self, host, lease):
        with self.stream:
            device_buffer = self._unaliased(host.to(self.device, non_blocking=True), host)
            event = self.stream.record_event()
        handle = TransferHandle(device_buffer, event, lease, self)
        with self._lock:
            self._in_flight.append(handle)
            self.transfers += 1
            self.bytes_sent += handle.nbytes
        return handle

    def _send_sync(self, host, lease):
        """No streams here, so the copy is done by the time this returns.

        The lease is still released through the guard rather than directly: one release path means
        one thing to get right, and it costs nothing when the event is always complete.
        """
        device_buffer = self._unaliased(host.to(self.device), host)
        handle = SyncTransferHandle(device_buffer, self.caps.event(), lease, self)
        with self._lock:
            self.transfers += 1
            self.bytes_sent += handle.nbytes
        handle._release_source()
        return handle

    # -- ordering ------------------------------------------------------------------------------

    def _await(self, handle):
        """Order the compute stream behind this transfer, then harvest what has finished.

        On the async path the CPU does not block: the wait is a device-side dependency. The
        allocator is also told that the compute stream now uses this buffer, because the buffer was
        allocated on the copy stream and without that the caching allocator is free to hand those
        bytes to something else while compute is still reading them -- the same class of bug as
        releasing a staging buffer early, one tier down.
        """
        if not self.is_async:
            return
        handle.event.wait()
        recorder = getattr(handle.device_buffer, "record_stream", None)
        if recorder is not None:
            try:
                recorder(torch.cuda.current_stream())
            except Exception:  # noqa: BLE001 - record_stream is an optimisation, not a requirement
                pass
        self.reclaim()

    def reclaim(self):
        """Release the staging buffers of every transfer that has finished. Never blocks."""
        with self._lock:
            pending = list(self._in_flight)
        still = []
        for handle in pending:
            if not handle.reclaim_if_complete():
                still.append(handle)
        with self._lock:
            # Anything sent while we were harvesting stays; only the ones we resolved are dropped.
            done = {id(h) for h in pending} - {id(h) for h in still}
            self._in_flight = [h for h in self._in_flight if id(h) not in done]
        return len(still)

    def drain(self):
        """Block until every outstanding transfer has finished and given its buffer back.

        For a generation boundary or shutdown: the pool cannot safely free a buffer that a copy is
        still reading, so someone has to wait, once, here.
        """
        with self._lock:
            pending, self._in_flight = list(self._in_flight), []
        for handle in pending:
            if not handle.is_complete():
                self.stalls += 1
            handle._release_source()
        return len(pending)

    # -- reporting -----------------------------------------------------------------------------

    def stats(self):
        with self._lock:
            in_flight = len(self._in_flight)
        return {
            "backend": self.caps.backend,
            "async": self.is_async,
            "transfers": self.transfers,
            "bytes_sent": self.bytes_sent,
            "in_flight": in_flight,
            "stalls": self.stalls,
        }

    def close(self):
        self.drain()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# The staging pool lives next door; re-exported so `rocketllm.streaming.transfer.HostStagingPool`
# keeps resolving for anything that already imports it from here.
from .staging import BufferLease, HostStagingPool  # noqa: E402  (placed after to avoid a cycle)

__all__ = ["WeightTransfer", "TransferHandle", "SyncTransferHandle", "HostStagingPool",
           "BufferLease"]
