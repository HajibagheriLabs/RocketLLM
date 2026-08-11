"""How many device bytes the weight cache may hold, measured continuously.

The number this publishes is the one every placement decision downstream is made against, and it
moves during a run. A generation grows its KV cache with every token; activations come and go;
another process can take a slice of the card at any moment. A budget computed once at load is
already wrong by the end of the first generation, and wrong in the dangerous direction -- it says
there is room that no longer exists.

So this measures. It does not model, and in particular it does not estimate the size of the KV
cache: whatever the cache, the activations and the rest of the machine are doing shows up in the
reading for free, including for an architecture nobody has written a size formula for yet. Adding an
estimator would replace a fact with a guess and then have to keep the guess current for every new
attention variant.

Two things make the reading usable. The first is the arithmetic in :meth:`VramBudget.measure`, which
adds back what the caching allocator is sitting on -- see the comment there, it is the part that
looks wrong and is not. The second is hysteresis: the raw reading jitters by the allocator's own
churn, and a cache that acted on every jog would evict and refetch a layer to learn nothing, at the
cost of a full streaming pass. So the published target only moves after the deviation persists.

What this publishes is what the OWNER may hold, not what the driver calls free, and the difference
matters once the machine is inside its reserve. Free memory alone floors at zero there, and a floor
destroys the sign: it cannot say how far over-committed the owner is, so nothing gives memory back,
and it reads every subsequent recovery as zero-to-zero, so nothing takes memory either. The cache
ends up frozen at whatever it happened to hold when the machine first went under, for the rest of
the generation. So the owner's own holdings are folded in BEFORE the floor -- see `reclaimable` --
and the floor is applied once, at the end, where it means "there is nothing left to keep".
"""
import dataclasses
import logging
import os
import threading
import time
import warnings
from collections import deque

import torch

from ..hw import caps
from ..hw.caps import announce_once, get_caps

log = logging.getLogger(__name__)

#: What torch says when a platform cannot do expandable segments. Matched rather than assumed from
#: the OS, because which platforms support it changes between torch releases.
_UNSUPPORTED_MARKER = "expandable_segments not supported"

_ALLOC_ENV = "PYTORCH_CUDA_ALLOC_CONF"
_EXPANDABLE = "expandable_segments:True"


@dataclasses.dataclass(frozen=True)
class AllocatorSetup:
    """What happened when we tried to configure the caching allocator."""

    #: "applied" | "unsupported" | "already_configured" | "too_late" | "not_applicable"
    status: str
    detail: str
    #: The value of PYTORCH_CUDA_ALLOC_CONF after the attempt, if any.
    setting: object = None

    @property
    def effective(self):
        return self.status == "applied"

    def to_dict(self):
        return dataclasses.asdict(self)


def configure_allocator(device=None, probe=True):
    """Ask for expandable segments before the first device allocation, where that applies.

    Under a streaming workload the allocator carves and releases a layer's worth of blocks hundreds
    of times per token, and without expandable segments each differently-sized layer strands the
    blocks the last one left behind. The setting is read once, when the CUDA context is created, so
    it has to be in the environment before anything touches the device -- which is why this is
    called at construction and not at first use.

    Whether it took effect is *measured*, not assumed: torch accepts the setting everywhere and then
    warns at the first allocation on platforms that cannot honour it, so the only truthful answer
    comes from watching for that warning. `probe=False` skips the tiny allocation that provokes it,
    at the cost of reporting "applied" without proof.
    """
    dev = caps.resolve_device(device) if not isinstance(device, torch.device) else device
    if caps.backend_of(dev) not in ("cuda", "rocm"):
        return AllocatorSetup("not_applicable",
                              f"{caps.backend_of(dev)} has no {_ALLOC_ENV}; nothing to configure")

    existing = os.environ.get(_ALLOC_ENV)
    if existing and "expandable_segments" in existing:
        return AllocatorSetup("already_configured",
                              f"{_ALLOC_ENV} already requests expandable segments; left alone",
                              existing)

    # Already initialised means the allocator has read its configuration and will not read it
    # again. Setting the variable now would look like it worked and change nothing, so say so.
    if torch.cuda.is_initialized():
        announce_once(
            "alloc-too-late",
            f"the device context was already created before the memory budget was built, so "
            f"{_ALLOC_ENV}={_EXPANDABLE} can no longer take effect. The allocator will fragment "
            f"more under streaming. Set it in the environment before importing torch.",
            logging.WARNING)
        return AllocatorSetup("too_late",
                              "the CUDA context was already initialised; the setting is read once "
                              "at context creation and cannot be changed afterwards", existing)

    os.environ[_ALLOC_ENV] = f"{existing},{_EXPANDABLE}" if existing else _EXPANDABLE
    if not probe:
        return AllocatorSetup("applied", f"{_ALLOC_ENV} set before the context was created; "
                                         "not verified", os.environ[_ALLOC_ENV])

    # One element is enough to create the context and provoke the warning if there is one.
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.zeros(1, device=dev)
        unsupported = any(_UNSUPPORTED_MARKER in str(w.message) for w in caught)
    except Exception as exc:  # noqa: BLE001 - a probe must never be the thing that fails a load
        return AllocatorSetup("applied",
                              f"{_ALLOC_ENV} set before the context was created; could not verify "
                              f"({exc})", os.environ[_ALLOC_ENV])

    if unsupported:
        announce_once(
            "alloc-unsupported",
            "this build of torch does not support expandable segments on this platform, so the "
            "allocator will fragment more under a streaming workload. Nothing is wrong with the "
            "run; peak device memory will simply sit higher than it needs to.",
            logging.INFO)
        return AllocatorSetup("unsupported",
                              "torch warned that expandable segments are unsupported on this "
                              "platform and ignored the request", os.environ[_ALLOC_ENV])
    return AllocatorSetup("applied", "expandable segments requested before the context was created "
                                     "and accepted", os.environ[_ALLOC_ENV])


@dataclasses.dataclass(frozen=True)
class BudgetSample:
    """One reading, with every term kept so a surprising budget can be taken apart."""

    at: float
    #: Bytes the driver calls free.
    free: object
    #: Bytes the caching allocator holds from the driver.
    reserved: object
    #: Bytes of that which are actually live.
    allocated: object
    #: Held-but-free allocator blocks: reserved - allocated.
    held: int
    reserve: int
    #: Bytes the owner already holds and could give back. Zero unless it said.
    reclaimable: int
    #: free + held - reserve, SIGNED. Negative means the machine is inside its reserve, and by how
    #: much -- which is the amount the owner is being asked to hand back.
    headroom: int
    #: max(0, headroom + reclaimable). What the owner may hold.
    usable: int
    #: True when the backend could not report the components and this is a conservative guess.
    estimated: bool
    #: The published budget at the time of this sample.
    target: int

    def to_dict(self):
        return dataclasses.asdict(self)


class VramBudget:
    """The live device budget, sampled at layer boundaries and published with hysteresis.

    `current()` is the latest raw measurement and `target()` is what callers should size against.
    They differ on purpose: the first tracks the machine, the second changes only when the machine
    has really changed, so a cache built against it is not asked to reorganise over noise.
    """

    def __init__(self, device=None, device_caps=None, profile=None, reserve_bytes=None,
                 hysteresis_bytes=None, hysteresis_samples=None, history=512, on_change=None,
                 configure_allocator_env=True, probe_allocator=True, hysteresis_ratio=None,
                 reclaimable=None):
        self.caps = device_caps if device_caps is not None else get_caps(device, announce=False)
        self.device = self.caps.device
        self.profile = profile

        #: Optional callable returning the bytes the owner already holds on the device. Settable
        #: after construction, because the budget is usually built before the thing it sizes.
        self.reclaimable = reclaimable

        # Set before the first allocation, which is why it happens in the constructor: by the time
        # anything asks for a budget the context usually exists, and then it is too late.
        self.allocator = (configure_allocator(self.device, probe=probe_allocator)
                          if configure_allocator_env else
                          AllocatorSetup("not_applicable", "allocator configuration was declined"))

        self.reserve_bytes = self._resolve("reserve_bytes", reserve_bytes, 0)
        # The band is a SHARE of the budget in play, not a fixed number of bytes. A fixed one has to
        # be sized against something, and the only thing available at probe time is the whole card
        # -- which is the wrong scale as soon as the budget is constrained. Measured on a 24GB card:
        # a band of 1693MB governing a budget of 507MB, so nothing could ever be published and a
        # real 66MB recovery never reached the pin plan. A share means the same thing at 4GB and at
        # 192GB, which is the property this needs and a byte count cannot have.
        #
        # An absolute value still wins where one is given, because that is the debugging override
        # for reproducing a suspected bad measurement.
        self._fixed_band = self._optional("budget_hysteresis_bytes", hysteresis_bytes)
        if self._fixed_band is not None:
            self._fixed_band = max(0, int(self._fixed_band))
        self.hysteresis_ratio = max(0.0, float(
            self._optional("budget_hysteresis_ratio", hysteresis_ratio) or 0.0))
        self._target = 0
        self.hysteresis_samples = max(1, self._resolve("budget_hysteresis_samples",
                                                       hysteresis_samples, 1))

        self.on_change = on_change
        self.history = deque(maxlen=history) if history else None
        self._lock = threading.Lock()
        self._streak = 0
        self._streak_sign = 0
        self._streak_low = None
        self.changes = 0

        # A budget nobody has sampled yet is not zero, it is unknown -- and zero would mean "cache
        # nothing", which is a decision this has not earned the right to make. So the first reading
        # is taken now and published unconditionally; hysteresis governs moves, not the start.
        first = self.measure()
        self._target = first.usable
        self._record(dataclasses.replace(first, target=self._target))

    # -- knob resolution -------------------------------------------------------------------------

    def _resolve(self, knob, override, fallback):
        """A knob's value: the caller's, else the profile's, else a stated degradation.

        `reserve` in particular is not a constant and not a percentage picked by hand -- it comes
        from what the allocator was measured doing on this machine. Without a profile there is no
        honest number, so the degradation is announced rather than papered over with one.
        """
        if override is not None:
            return int(override)
        if self.profile is not None:
            derivation = self.profile.derived.get(knob)
            if derivation is not None:
                return int(derivation.value)
        announce_once(
            f"budget-noprofile-{knob}",
            f"no hardware profile was available to supply {knob}, so the device budget falls back "
            f"to {fallback}. It will still track free memory, but without the machine's own "
            f"measurements behind it.", logging.INFO)
        return fallback

    def _optional(self, knob, override):
        """A knob that may legitimately be absent, with no complaint when it is.

        Separate from :meth:`_resolve` because a missing `reserve` is a degradation worth announcing
        and a missing absolute hysteresis band is not -- the band is normally a share, and only some
        configurations pin it to a byte count.
        """
        if override is not None:
            return override
        if self.profile is not None:
            derivation = self.profile.derived.get(knob)
            if derivation is not None:
                return derivation.value
        return None

    def _reclaimable(self):
        """Bytes the owner already holds, or zero when nobody said.

        This is a measurement like any other here, so it may not be the thing that fails a sample:
        the owner is arbitrary caller code and may be mid-teardown when a sample lands.
        """
        if self.reclaimable is None:
            return 0
        try:
            return max(0, int(self.reclaimable()))
        except Exception:  # noqa: BLE001 - a budget that cannot measure still has to publish
            log.debug("the reclaimable callable raised; counting it as zero", exc_info=True)
            return 0

    # -- the band ----------------------------------------------------------------------------------

    def scale(self):
        """The size of the pool this budget governs: the largest usable figure seen recently.

        Choosing this basis took two wrong answers first, and both are worth recording.

        The instantaneous usable figure is wrong because the cache this budget sizes expands until
        free memory is gone. Usable therefore tends toward zero during a healthy run, and a band
        proportional to it tends to zero with it, which leaves every allocator jog publishing -- the
        opposite of hysteresis.

        Usable plus what the process has allocated is wrong because it counts allocations that have
        nothing to do with this budget: any other tensor the process is holding inflates the band,
        and the band then suppresses exactly the changes it exists to let through.

        The high-water mark over the retained history has neither problem. It is the largest budget
        this run has actually been given, it is composed only of readings already taken, and because
        the history is bounded it decays: a card that really has permanently lost memory to another
        process stops being measured against the pool it used to have.

        The first of those objections is also why the owner's holdings belong inside `usable`
        rather than being added by the caller afterwards. A cache that has filled the card would
        otherwise drive every reading to zero, take the band with it, and start publishing on noise
        at exactly the moment residency is worth the most.
        """
        seen = max((reading.usable for reading in self.history), default=0) if self.history else 0
        return max(seen, self._target)

    def band_for(self):
        """How far a reading must sit from the published target before the move is treated as real.

        Proportional to the pool in play, so it carries the same meaning on a 4GB card and a 192GB
        one. The share comes from the profile, where it is the larger of a policy floor and the
        allocator fragmentation actually measured on this machine -- the size of the noise being
        rejected. An absolute band, where one was given, wins over all of it.
        """
        if self._fixed_band is not None:
            return self._fixed_band
        return int(self.scale() * self.hysteresis_ratio)

    @property
    def hysteresis_bytes(self):
        """The band as it currently stands, for reporting."""
        return self.band_for()

    # -- measurement -----------------------------------------------------------------------------

    def measure(self):
        """One reading of what the cache may hold, without touching the published target.

        The arithmetic is:

            free_from_driver + (memory_reserved() - memory_allocated()) - reserve + reclaimable

        The middle term is the part that looks like a mistake and is not. DO NOT "SIMPLIFY" IT AWAY.
        torch's caching allocator does not hand memory back to the driver when a tensor is freed --
        it keeps the block to serve the next allocation without a synchronising driver call, which
        is precisely why streaming a layer per token is affordable at all. The driver still counts
        every one of those blocks as in use, so `mem_get_info` reports them as unavailable. They
        are not: this process can allocate into them immediately. `memory_reserved - memory_allocated`
        is exactly that pool, and adding it back is the difference between a cache that fills the
        card and one that gives up with gigabytes going spare.

        The last term is what the owner already holds. Those bytes are not free -- they are
        allocated, by the owner, for exactly the purpose being sized -- so they belong in the answer
        to "how much may you hold", and they have to be added before the floor rather than after it.
        Added after, as this used to do, the floor flattens the whole region below the reserve to
        zero and the answer degenerates to "however much you are already holding": the owner can
        then only ever ratchet downwards, and memory freed while it is down there is invisible.

        Backends that cannot report the components fall back to a conservative reading -- what the
        backend calls free, with nothing added back -- and say so, once, rather than pretending to
        an accuracy they do not have.
        """
        report = self.caps.memory(self.reserve_bytes)

        held = 0
        if report.reserved is not None and report.allocated is not None:
            held = max(0, int(report.reserved) - int(report.allocated))
        elif report.estimated:
            announce_once(
                f"budget-estimated-{self.caps.backend}",
                f"the {self.caps.backend} backend does not report allocator reserved/allocated "
                f"totals, so the device budget is the conservative reading -- free memory minus "
                f"reserve, with no allowance for blocks the allocator is holding. It will "
                f"under-claim rather than over-claim.", logging.INFO)

        # Recomposed from the components rather than read off `report.budget`, which every backend
        # has already floored. The floor is the thing being avoided here, and the components it was
        # computed from are all on the report, so this loses nothing: report.budget is exactly
        # max(0, headroom) on every backend, and there is a test that says so.
        headroom = (int(report.budget) if report.free is None
                    else int(report.free) + held - self.reserve_bytes)
        reclaimable = self._reclaimable()

        return BudgetSample(
            at=time.monotonic(), free=report.free, reserved=report.reserved,
            allocated=report.allocated, held=held, reserve=self.reserve_bytes,
            reclaimable=reclaimable, headroom=headroom,
            usable=max(0, headroom + reclaimable), estimated=bool(report.estimated),
            target=getattr(self, "_target", 0))

    def sample(self):
        """Measure, apply hysteresis, and publish a new target if the shift has persisted.

        Cheap enough for every layer boundary: one driver query plus two allocator counters, no
        allocation and no synchronisation.
        """
        reading = self.measure()
        with self._lock:
            previous = self._target
            deviation = reading.usable - previous
            sign = (deviation > 0) - (deviation < 0)
            # Measured against the band for the budget being operated on, not a fixed number of
            # bytes, so the same relative move is judged the same way at any scale.
            band = self.band_for()

            if abs(deviation) <= band:
                # Inside the noise band. Not a move, and it breaks any run that was building.
                self._streak = 0
                self._streak_sign = 0
                self._streak_low = None
            else:
                if sign != self._streak_sign:
                    # A reading that overshot in the other direction is not a continuation of the
                    # trend, it is the jitter this exists to reject. Start the count again.
                    self._streak = 0
                    self._streak_low = None
                self._streak += 1
                self._streak_sign = sign
                self._streak_low = (reading.usable if self._streak_low is None
                                    else min(self._streak_low, reading.usable))

            changed = None
            if self._streak >= self.hysteresis_samples:
                # Publish the most conservative reading of the run, in both directions. When the
                # budget is shrinking that is the only safe choice; when it is growing it merely
                # claims the new room a sample later than it could have.
                self._target = self._streak_low
                self._streak = 0
                self._streak_sign = 0
                self._streak_low = None
                self.changes += 1
                changed = (previous, self._target)

            reading = dataclasses.replace(reading, target=self._target)
            self._record(reading)

        # Outside the lock: a callback is the caller's code and may do anything, including calling
        # back into this object.
        if changed is not None:
            log.debug("device budget moved %d -> %d bytes", changed[0], changed[1])
            if self.on_change is not None:
                self.on_change(changed[0], changed[1], reading)
        return reading

    def _record(self, reading):
        if self.history is not None:
            self.history.append(reading)

    # -- what callers read -----------------------------------------------------------------------

    def current(self):
        """The latest raw measurement, noise and all."""
        return self.history[-1].usable if self.history else self.measure().usable

    def target(self):
        """The published budget. Size caches against this, not against `current()`."""
        return self._target

    def reset(self):
        """Forget the hysteresis state and republish whatever is true now.

        For a generation boundary, where the KV cache has just been dropped: the budget is about to
        jump by its whole size, and making that wait for a streak would leave the cache sized for a
        context that no longer exists.
        """
        with self._lock:
            reading = self.measure()
            previous, self._target = self._target, reading.usable
            self._streak = 0
            self._streak_sign = 0
            self._streak_low = None
            reading = dataclasses.replace(reading, target=self._target)
            self._record(reading)
        if previous != self._target:
            self.changes += 1
            if self.on_change is not None:
                self.on_change(previous, self._target, reading)
        return reading

    # -- reporting -------------------------------------------------------------------------------

    def trace(self):
        """The history as plain dicts, for the benchmark harness and bug reports."""
        return [reading.to_dict() for reading in (self.history or ())]

    def summary(self):
        readings = list(self.history or ())
        usable = [r.usable for r in readings]
        latest = readings[-1] if readings else None
        return {
            "backend": self.caps.backend,
            "device": str(self.device),
            "reserve_bytes": self.reserve_bytes,
            "reclaimable_bytes": latest.reclaimable if latest is not None else 0,
            "headroom_bytes": latest.headroom if latest is not None else 0,
            "hysteresis_bytes": self.hysteresis_bytes,
            "hysteresis_ratio": self.hysteresis_ratio,
            "hysteresis_scale_bytes": self.scale(),
            "hysteresis_samples": self.hysteresis_samples,
            "target_bytes": self._target,
            "samples": len(readings),
            "changes": self.changes,
            "min_usable_bytes": min(usable) if usable else None,
            "max_usable_bytes": max(usable) if usable else None,
            "estimated": bool(readings and readings[-1].estimated),
            "allocator": self.allocator.to_dict(),
        }

    def __repr__(self):
        return (f"<VramBudget {self.caps.backend} target={self._target} "
                f"reserve={self.reserve_bytes} +/-{self.hysteresis_bytes}x{self.hysteresis_samples}>")
