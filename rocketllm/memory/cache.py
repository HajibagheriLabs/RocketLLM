"""The tiered weight cache: device, host, storage.

One entry per logical unit the engine streams -- a dense decoder layer, or a single MoE expert --
keyed by ``(layer_idx, kind)`` and sized in PACKED bytes, because packed bytes are what actually
cross the link and occupy the card. Expanded size is a property of the running device, not of the
cache, and mixing the two into the sizing arithmetic is how a cache ends up confidently over-filling
a GPU.

Three tiers, and the middle one is the point. A weight evicted from the device does not have to fall
all the way back to storage: host RAM serves it at PCIe speed instead of disk speed, which on a
storage-bound machine is two orders of magnitude. So eviction moves entries device -> host, and only
drops them entirely when the host tier is full too. On a machine with little free RAM that tier is
near-zero, and that has to work: it degrades to device -> storage, which is exactly what the cache
did before the tier existed.

The replacement policies are deliberately different for the two kinds of entry, and the asymmetry is
load-bearing. See :meth:`TieredWeightCache._dense_victims` for why dense layers must not use LRU.
"""
import dataclasses
import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)

#: Entry kinds. "expert:N" carries the expert ordinal because experts within a layer have wildly
#: different popularity and are cached independently.
KIND_DENSE = "dense"


def expert_kind(expert_index):
    return f"expert:{expert_index}"


def is_expert(kind):
    return str(kind).startswith("expert:")


@dataclasses.dataclass
class CacheEntry:
    """One cached unit, wherever it currently lives."""

    key: tuple
    packed_bytes: int
    #: "device" | "host"; an entry that is in neither is not in the cache at all.
    tier: str
    payload: object = None
    #: Live users. An entry with any is mid-forward and must not move.
    refcount: int = 0
    pinned: bool = False
    #: Popularity, for experts. Halved periodically so early hotness decays.
    uses: float = 0.0
    #: Monotonic insertion ordinal, for the dense window's FIFO order.
    inserted: int = 0

    @property
    def in_use(self):
        return self.refcount > 0

    @property
    def evictable(self):
        return not self.pinned and not self.in_use


class _Tier:
    """A byte-capped set of entries. Zero capacity is a supported configuration, not an error."""

    def __init__(self, name, capacity_bytes):
        self.name = name
        self.capacity = max(0, int(capacity_bytes or 0))
        self.entries = OrderedDict()
        self.bytes = 0

    def add(self, entry):
        self.entries[entry.key] = entry
        self.bytes += entry.packed_bytes
        entry.tier = self.name

    def remove(self, key):
        entry = self.entries.pop(key, None)
        if entry is not None:
            self.bytes -= entry.packed_bytes
        return entry

    def fits(self, nbytes):
        return self.bytes + nbytes <= self.capacity

    @property
    def free(self):
        return max(0, self.capacity - self.bytes)


class TieredWeightCache:
    """Device / host / storage residency for streamed weights.

    Callers use :meth:`acquire` and :meth:`release` around a module's forward. An acquired entry is
    refcounted and cannot be evicted underneath the forward that is reading it, which is the only
    hard invariant here -- everything else is a performance decision.
    """

    def __init__(self, fetch, sizer, device_bytes=0, host_bytes=0, pinned=(), window=1,
                 aging_interval=4096, profile=None, to_device=None, to_host=None,
                 discard=None, sequence=None, prefetch_workers=0):
        #: Read a key's packed payload from storage. The slow path, and the one being avoided.
        self._fetch = fetch
        #: Packed bytes for a key. In the engine this comes from the quant registry's PackedWeight,
        #: which is what knows that a 4-bit payload costs its packed size and not its expanded one.
        self._sizer = sizer
        #: Optional tier transfer hooks, so the engine can move real tensors and the tests can not.
        self._to_device = to_device or (lambda payload: payload)
        self._to_host = to_host or (lambda payload: payload)
        #: Called when an entry leaves the cache entirely, so the owner can unbind or free it.
        self._discard = discard or (lambda payload: None)
        #: Given a key, the keys that will be wanted next. This is what lets the cache own
        #: lookahead instead of the caller keeping a one-slot prefetch of its own.
        self._sequence = sequence

        self.device = _Tier("device", device_bytes)
        self.host = _Tier("host", self._knob(profile, "host_cache_bytes", host_bytes))
        self.aging_interval = max(1, int(self._knob(profile, "expert_aging_interval",
                                                    aging_interval)))

        self.pinned = set(pinned)
        #: How many dense layers may sit on the device at once, over and above the pinned subset.
        self.window = max(1, int(window))

        self._entries = {}
        self._lock = threading.RLock()
        self._clock = 0
        self._expert_accesses = 0

        # Lookahead. Only the storage read runs on a worker: moving a layer onto the device binds
        # parameters on the model, and doing that from another thread while a forward is running is
        # a race the engine would never recover from. So the worker reads, the caller places.
        # An explicit width wins over the profile's: the caller passing one is a debugging override
        # and has to be able to override the measurement it is there to question.
        workers = int(prefetch_workers or self._knob(profile, "io_workers", 0) or 0)
        self._executor = (ThreadPoolExecutor(max_workers=workers,
                                             thread_name_prefix="rocketllm-prefetch")
                          if workers > 0 and self._sequence is not None else None)
        self._pending = {}

        self.stats = {
            "hits_device": 0, "hits_host": 0, "misses": 0,
            "evicted_to_host": 0, "evicted_to_storage": 0, "host_evictions": 0,
            "fetches": 0, "promotions": 0, "agings": 0,
            "rejected_too_large": 0, "prefetches": 0, "prefetch_hits": 0,
        }

    @staticmethod
    def _knob(profile, name, fallback):
        if profile is not None:
            derivation = profile.derived.get(name)
            if derivation is not None:
                return int(derivation.value)
        return fallback

    @classmethod
    def for_model(cls, fetch, sizer, profile, largest_layer_bytes, device_bytes=None,
                 pinned=(), **kwargs):
        """Build with the window width the machine and the model imply.

            W = clamp(window_budget / max_layer_packed_bytes, 1, window_max)

        Never below one: a cache that cannot hold a single layer cannot run a forward at all, and
        the honest failure for that is an explicit error from the engine naming the smallest
        configuration that would work -- not a window of zero that deadlocks.
        """
        window_budget = cls._knob(profile, "window_budget_bytes", 0)
        largest = max(1, int(largest_layer_bytes or 1))
        window = max(1, window_budget // largest)
        if profile is not None and hasattr(profile, "window_max"):
            window = max(1, min(window, profile.window_max(largest)))
        return cls(fetch=fetch, sizer=sizer, profile=profile, window=window,
                   device_bytes=device_bytes if device_bytes is not None
                   else cls._knob(profile, "usable_device_bytes", 0),
                   pinned=pinned, **kwargs)

    # -- residency ---------------------------------------------------------------------------

    def acquire(self, key):
        """Get a key's payload, from the fastest tier that has it, and pin it for the caller.

        Every path through here ends with the entry on the device and its refcount raised. Release
        it when the module has run.
        """
        with self._lock:
            entry = self._entries.get(key)

            if entry is not None and entry.tier == "device":
                self.stats["hits_device"] += 1
                self._touch(entry)
                entry.refcount += 1
                return self._resolve(entry)

            if entry is not None and entry.tier == "transient":
                # Still bound from a claim that has not been released yet. No tier holds it, but the
                # payload is live and re-reading it would strand the copy already in use.
                self.stats["hits_device"] += 1
                entry.refcount += 1
                return self._resolve(entry)

            if entry is not None and entry.tier == "host":
                # Served from RAM over the link rather than re-read from disk. On a storage-bound
                # machine this is the difference the host tier exists to make.
                self.stats["hits_host"] += 1
                self.host.remove(key)
                self._make_room(entry.packed_bytes, exclude=key)
                entry.payload = self._to_device(entry.payload)
                self.device.add(entry)
                self.stats["promotions"] += 1
                self._touch(entry)
                entry.refcount += 1
                return self._resolve(entry)

            self.stats["misses"] += 1
            payload = self._take_pending(key)
            if payload is None:
                payload = self._fetch(key)
                self.stats["fetches"] += 1
            else:
                self.stats["prefetch_hits"] += 1
            entry = CacheEntry(key=key, packed_bytes=int(self._sizer(key)), tier="device",
                               payload=self._to_device(payload),
                               pinned=key in self.pinned)
            self._make_room(entry.packed_bytes, exclude=key)
            self._admit(entry)
            self._touch(entry)
            entry.refcount += 1
            return self._resolve(entry)

    # -- lookahead -----------------------------------------------------------------------------

    def prefetch_window(self, key):
        """Start reading the layers that come after `key`, up to the window's width.

        Only the storage read is started here, on a worker; the placement happens on whichever
        thread calls acquire. That is deliberate -- see the note where the executor is built.

        Anything already resident, already in flight, or pinned is skipped, so this is cheap to
        call on every layer boundary and does no work at all once the model fits.
        """
        if self._executor is None or self._sequence is None:
            return 0
        started = 0
        for upcoming in self._sequence(key, self.window):
            with self._lock:
                if upcoming in self._entries or upcoming in self._pending:
                    continue
                self._pending[upcoming] = self._executor.submit(self._prefetch_one, upcoming)
            started += 1
        return started

    def _prefetch_one(self, key):
        self.stats["fetches"] += 1
        self.stats["prefetches"] += 1
        return self._fetch(key)

    def _take_pending(self, key):
        """Collect a prefetched payload, waiting for it if the read is still running."""
        with self._lock:
            future = self._pending.pop(key, None)
        if future is None:
            return None
        try:
            return future.result()
        except Exception:  # noqa: BLE001 - a failed prefetch must fall back to a direct read
            log.debug("prefetch of %s failed; reading it directly", key, exc_info=True)
            return None

    def _drop_pending(self):
        with self._lock:
            pending, self._pending = list(self._pending.values()), {}
        for future in pending:
            future.cancel()
        for future in pending:
            if not future.cancelled():
                try:
                    future.result()
                except Exception:  # noqa: BLE001 - draining, not using the result
                    pass

    def _resolve(self, entry):
        """Settle a payload that is still arriving before handing it to the caller.

        The streaming path returns a TransferHandle rather than a tensor: the copy has been issued
        on the copy stream and has not necessarily landed. Resolving it queues a wait on the compute
        stream -- a device-side dependency, so this does not block the CPU -- and gives back the
        staged host buffer once the copy is known to have finished. That is what makes the transfer
        overlap the previous layer's compute instead of costing wall-clock time here.

        Payloads that are already plain values, which is everything in the tests and every non-
        streaming caller, have no resolve() and pass straight through.
        """
        resolver = getattr(entry.payload, "resolve", None)
        if resolver is None:
            return entry.payload
        entry.payload = resolver()
        return entry.payload

    def release(self, key):
        """Give up a claim on an entry. It becomes evictable once nobody holds it.

        A transient entry -- one admitted when there was nowhere to keep it -- is not merely
        evictable at this point, it is finished, so this is where it is handed back. Without that
        the pure-streaming configuration never releases anything: the entry is untracked, so nothing
        else can ever discard it, and the weights stay bound for the rest of the run. That is the
        exact opposite of streaming, and it is the configuration a device with no spare memory gets.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            if entry.refcount > 0:
                entry.refcount -= 1
            if entry.tier != "transient" or entry.refcount > 0:
                return
            self._entries.pop(key, None)
            payload, entry.payload = entry.payload, None
        self._discard(payload)

    def _admit(self, entry):
        self._clock += 1
        entry.inserted = self._clock
        self._entries[entry.key] = entry
        if self.device.capacity <= 0 or entry.packed_bytes > self.device.capacity:
            # Nowhere to keep it. The caller still gets the payload -- the forward must run -- it
            # simply will not be here next time. This is the pure-streaming configuration, and it
            # is a supported one.
            #
            # It stays tracked until the caller releases it, even though no tier holds it. Dropping
            # it here instead would leave the payload with no owner: release() would find nothing,
            # and the weights would never be handed back.
            if entry.packed_bytes > self.device.capacity and self.device.capacity > 0:
                self.stats["rejected_too_large"] += 1
            entry.tier = "transient"
            return
        self.device.add(entry)
        if not is_expert(entry.key[1]):
            self._enforce_window()

    def _enforce_window(self):
        """Hold the unpinned dense layers on the device to the prefetch window's width.

        The pinned subset is not part of the window and is not counted against it: those layers are
        resident for the whole run by decision, and the window is the pipeline of layers moving
        through on top of them. Draining it from the front is the FIFO half of the policy -- see
        :meth:`_dense_victims` for why it must not become LRU.
        """
        while True:
            resident = [e for e in self.device.entries.values()
                        if not is_expert(e.key[1]) and not e.pinned]
            if len(resident) <= self.window:
                return
            evictable = [e for e in resident if e.evictable]
            if not evictable:
                return
            self._demote(min(evictable, key=lambda e: e.inserted))

    def _touch(self, entry):
        """Record a use. Only experts keep a popularity count; dense entries are FIFO by design."""
        if not is_expert(entry.key[1]):
            return
        entry.uses += 1.0
        self._expert_accesses += 1
        if self._expert_accesses % self.aging_interval == 0:
            self._age()

    def _age(self):
        """Halve every expert's count.

        Without this an expert that was hot in the first hundred tokens outranks one that is hot
        now for the rest of the run, because a raw count only ever grows. Halving keeps the order
        while letting recent behaviour overtake old behaviour.
        """
        for entry in self._entries.values():
            if is_expert(entry.key[1]):
                entry.uses *= 0.5
        self.stats["agings"] += 1

    # -- eviction ------------------------------------------------------------------------------

    def _make_room(self, nbytes, exclude=None):
        """Free `nbytes` on the device, demoting to host where the host has room."""
        if self.device.capacity <= 0:
            return
        while self.device.bytes + nbytes > self.device.capacity:
            victim = self._next_victim(exclude)
            if victim is None:
                # Everything left is pinned or in use. Correct behaviour is to proceed: the
                # allocator will take the hit, and the alternative is refusing to run a forward.
                return
            self._demote(victim)

    def _next_victim(self, exclude=None):
        """The entry whose residency is worth least right now.

        Experts go before dense layers. A dense layer on the device is one the cyclic scan will
        reach again within a fixed number of steps; a cold expert may not be routed to for
        thousands of tokens, so its bytes are the cheapest on the card.
        """
        for candidate in self._expert_victims():
            if candidate.key != exclude:
                return candidate
        for candidate in self._dense_victims():
            if candidate.key != exclude:
                return candidate
        return None

    def _expert_victims(self):
        """Experts, least popular first -- LFU, with the aging applied in :meth:`_age`.

        LFU is right here and wrong for dense layers, and the asymmetry is deliberate. Expert
        popularity is heavily skewed: a token routes to a handful out of hundreds, and the same
        handful keeps coming up, so a frequency count predicts the next access well.
        """
        experts = [e for e in self.device.entries.values()
                   if is_expert(e.key[1]) and e.evictable]
        return sorted(experts, key=lambda e: (e.uses, e.inserted))

    def _dense_victims(self):
        """Dense layers, oldest admitted first -- FIFO, and deliberately NOT LRU.

        Decode walks the decoder layers cyclically: 0, 1, ... L-1, 0, 1, ... L-1. When the cache
        holds fewer layers than the model has, LRU is precisely the wrong policy for that pattern.
        The least recently used layer is always the one furthest back in the scan, which is the one
        the cycle is about to come round to next -- so LRU evicts exactly the entry that is needed
        soonest, every single time, and the hit rate collapses to roughly zero. The classic
        pathological case, and it does not announce itself: the cache still works, it is just
        useless.

        FIFO does not fix the cyclic scan either -- nothing does, for a scan larger than the cache
        -- but it is the right structure for what actually buys hits here: a statically pinned
        subset that is never evicted, plus a prefetch window admitted in the order the scan will
        read it. The window is a pipeline, and a pipeline is drained from the front.

        If you are here to "simplify" this to LRU, the regression test in tests/test_cache_policy.py
        exists specifically to stop you.
        """
        dense = [e for e in self.device.entries.values()
                 if not is_expert(e.key[1]) and e.evictable]
        return sorted(dense, key=lambda e: e.inserted)

    def _demote(self, entry):
        """Device -> host if the host will take it, otherwise out of the cache entirely."""
        self.device.remove(entry.key)
        if self.host.capacity > 0:
            self._make_host_room(entry.packed_bytes)
            if self.host.fits(entry.packed_bytes):
                entry.payload = self._to_host(entry.payload)
                self.host.add(entry)
                self.stats["evicted_to_host"] += 1
                return
        self._entries.pop(entry.key, None)
        self._discard(entry.payload)
        entry.payload = None
        entry.tier = "storage"
        self.stats["evicted_to_storage"] += 1

    def _make_host_room(self, nbytes):
        while self.host.bytes + nbytes > self.host.capacity:
            victims = [e for e in self.host.entries.values() if e.evictable]
            if not victims:
                return
            # Same asymmetry as the device tier: cold experts first, then the oldest dense entry.
            victims.sort(key=lambda e: (0 if is_expert(e.key[1]) else 1, e.uses, e.inserted))
            victim = victims[0]
            self.host.remove(victim.key)
            self._entries.pop(victim.key, None)
            victim.payload = None
            victim.tier = "storage"
            self._discard(victim.payload)
            self.stats["host_evictions"] += 1

    # -- pinning -------------------------------------------------------------------------------

    def apply_plan(self, plan):
        """Adopt a new pin plan, keeping what both plans pin and releasing what only the old did."""
        with self._lock:
            new = set(plan.pinned) if hasattr(plan, "pinned") else set(plan)
            for key in self.pinned - new:
                entry = self._entries.get(key)
                if entry is not None:
                    entry.pinned = False
            for key in new:
                entry = self._entries.get(key)
                if entry is not None:
                    entry.pinned = True
            self.pinned = new
            return self.pinned

    def resize_device(self, device_bytes):
        """Follow the live budget. Shrinking evicts down to the new capacity immediately."""
        with self._lock:
            self.device.capacity = max(0, int(device_bytes or 0))
            self._make_room(0)

    # -- reporting -----------------------------------------------------------------------------

    def clear(self, keep_pinned=False):
        """Drop everything the cache is holding and unbind it from whoever owns it.

        For a generation boundary. An in-use entry is still not touched -- releasing weights a
        forward is reading would be the same bug eviction is careful about -- so anything acquired
        and not yet released survives, and there should be none of those between generations.
        """
        self._drop_pending()
        with self._lock:
            for key, entry in list(self._entries.items()):
                if entry.in_use or (keep_pinned and entry.pinned):
                    continue
                self.device.remove(key)
                self.host.remove(key)
                self._entries.pop(key, None)
                self._discard(entry.payload)
                entry.payload = None
                entry.tier = "storage"

    def close(self):
        self._drop_pending()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self.clear()

    def tier_of(self, key):
        entry = self._entries.get(key)
        return entry.tier if entry is not None else "storage"

    def report(self):
        """Hit and miss counts by tier, for the benchmark harness."""
        hits = self.stats["hits_device"] + self.stats["hits_host"]
        total = hits + self.stats["misses"]
        return dict(
            self.stats,
            entries=len(self._entries),
            device_entries=len(self.device.entries),
            host_entries=len(self.host.entries),
            device_bytes=self.device.bytes,
            host_bytes=self.host.bytes,
            device_capacity=self.device.capacity,
            host_capacity=self.host.capacity,
            window=self.window,
            pinned=len(self.pinned),
            hit_rate=(hits / total) if total else 0.0,
            device_hit_rate=(self.stats["hits_device"] / total) if total else 0.0,
            host_hit_rate=(self.stats["hits_host"] / total) if total else 0.0,
        )
