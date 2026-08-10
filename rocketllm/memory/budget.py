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
    #: free + held - reserve, floored at zero. What the cache may hold.
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
                 configure_allocator_env=True, probe_allocator=True):
        self.caps = device_caps if device_caps is not None else get_caps(device, announce=False)
        self.device = self.caps.device
        self.profile = profile

        # Set before the first allocation, which is why it happens in the constructor: by the time
        # anything asks for a budget the context usually exists, and then it is too late.
        self.allocator = (configure_allocator(self.device, probe=probe_allocator)
                          if configure_allocator_env else
                          AllocatorSetup("not_applicable", "allocator configuration was declined"))

        self.reserve_bytes = self._resolve("reserve_bytes", reserve_bytes, 0)
        self.hysteresis_bytes = max(0, self._resolve("budget_hysteresis_bytes", hysteresis_bytes, 0))
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

    # -- measurement -----------------------------------------------------------------------------

    def measure(self):
        """One reading of what the cache may hold, without touching the published target.

        The arithmetic is:

            free_from_driver + (memory_reserved() - memory_allocated()) - reserve

        The middle term is the part that looks like a mistake and is not. DO NOT "SIMPLIFY" IT AWAY.
        torch's caching allocator does not hand memory back to the driver when a tensor is freed --
        it keeps the block to serve the next allocation without a synchronising driver call, which
        is precisely why streaming a layer per token is affordable at all. The driver still counts
        every one of those blocks as in use, so `mem_get_info` reports them as unavailable. They
        are not: this process can allocate into them immediately. `memory_reserved - memory_allocated`
        is exactly that pool, and adding it back is the difference between a cache that fills the
        card and one that gives up with gigabytes going spare.

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

        return BudgetSample(
            at=time.monotonic(), free=report.free, reserved=report.reserved,
            allocated=report.allocated, held=held, reserve=self.reserve_bytes,
            usable=max(0, int(report.budget)), estimated=bool(report.estimated),
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

            if abs(deviation) <= self.hysteresis_bytes:
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
        return {
            "backend": self.caps.backend,
            "device": str(self.device),
            "reserve_bytes": self.reserve_bytes,
            "hysteresis_bytes": self.hysteresis_bytes,
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
