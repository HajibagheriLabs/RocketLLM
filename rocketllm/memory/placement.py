"""What to keep resident on the device, ranked by what residency actually buys.

Pinning is the only way to stop paying for a weight twice. Every byte kept on the device is a byte
not read from storage on the next token, so the question is never "what is big" or "what is first",
it is *how many storage bytes does one resident byte save per token*. That ratio is
accesses-per-token divided by packed size, and filling the budget greedily by it is what this
module does.

For a mixture-of-experts the ratio alone is not enough, because it would happily spend the whole
budget on a handful of enormous experts that happen to be popular and leave the attention block --
which every single token needs -- streaming from disk. So candidates are ranked inside priority
classes and no amount of frequency-per-byte promotes one past a class boundary:

  1. attention, norms and the router: touched by every token, and small
  2. shared / always-on experts: touched by every token, and large
  3. routed experts, by measured popularity

Pinning whole layers instead is strictly worse for an MoE and is not offered here. A layer's experts
are most of its bytes and a token routes to a few of them, so pinning the layer spends the budget on
weights that will not be read, and evicts the attention block that will.

The plan has to be sane across the whole range of machines this runs on: one that fits the entire
model, one that fits a fraction of it, and one whose pin budget computes to zero. The last is not a
failure -- it is pure streaming, it is a supported configuration, and it must produce an empty plan
rather than an error.
"""
import dataclasses
import logging

log = logging.getLogger(__name__)

#: Priority classes, lowest first. A class boundary is absolute: a wildly popular expert never
#: displaces the attention block that every token reads.
CLASS_ALWAYS = 0
CLASS_SHARED = 1
CLASS_ROUTED = 2

CLASS_NAMES = {
    CLASS_ALWAYS: "always-on (attention, norms, router, dense layers)",
    CLASS_SHARED: "shared experts",
    CLASS_ROUTED: "routed experts",
}


@dataclasses.dataclass(frozen=True)
class PinCandidate:
    """One thing that could be kept resident, and what keeping it would save.

    `accesses_per_token` is measured where the engine can measure it -- routed experts have wildly
    skewed popularity and only a running count knows it -- and is 1.0 for anything every token
    reads by construction.
    """

    key: object
    packed_bytes: int
    priority: int = CLASS_ALWAYS
    accesses_per_token: float = 1.0
    label: str = ""

    @property
    def savings_per_resident_byte(self):
        """Storage bytes saved per token, per byte of device memory spent keeping it.

        This is the whole ranking. A weight read twice per token is worth twice as much resident as
        one read once, and a weight half the size is worth twice as much per byte it occupies.
        """
        if self.packed_bytes <= 0:
            return 0.0
        return self.accesses_per_token / self.packed_bytes

    @property
    def bytes_saved_per_token(self):
        return self.accesses_per_token * self.packed_bytes


@dataclasses.dataclass(frozen=True)
class PinPlan:
    """The set of keys to keep resident, and why the rest did not make it."""

    pinned: tuple
    bytes_pinned: int
    budget_bytes: int
    #: (key, reason) for everything considered and rejected, in ranked order.
    skipped: tuple
    bytes_saved_per_token: float
    by_class: dict

    @property
    def is_pure_streaming(self):
        return not self.pinned

    def __contains__(self, key):
        return key in self.pinned_set

    @property
    def pinned_set(self):
        return frozenset(self.pinned)

    def explain(self):
        if self.is_pure_streaming:
            return (f"nothing is pinned: the pin budget is {self.budget_bytes} bytes. Every weight "
                    f"is streamed from storage on every token, which is correct and slow.")
        parts = [f"{count} {CLASS_NAMES.get(cls, cls)}" for cls, count in sorted(self.by_class.items())
                 if count]
        return (f"pinned {len(self.pinned)} entries ({self.bytes_pinned} of {self.budget_bytes} "
                f"budget bytes): {', '.join(parts)}; saves {self.bytes_saved_per_token:.0f} storage "
                f"bytes per token")

    def to_dict(self):
        return {
            "pinned": list(self.pinned),
            "bytes_pinned": self.bytes_pinned,
            "budget_bytes": self.budget_bytes,
            "skipped": len(self.skipped),
            "bytes_saved_per_token": self.bytes_saved_per_token,
            "by_class": dict(self.by_class),
            "pure_streaming": self.is_pure_streaming,
        }


def rank(candidates):
    """Candidates in the order they should be offered the budget.

    Class first, then savings per resident byte. Ties break towards the smaller weight, because two
    weights of equal value per byte are better taken as the one that leaves more budget behind, and
    finally on the key so a plan is reproducible rather than dependent on dict ordering.
    """
    return sorted(candidates,
                  key=lambda c: (c.priority, -c.savings_per_resident_byte, c.packed_bytes,
                                 str(c.key)))


def plan_pins(candidates, budget_bytes):
    """Fill the pin budget greedily, best value per resident byte first.

    Greedy rather than exact: this is a knapsack, an optimal solution is not worth computing on
    every budget change, and the ranking is by value density, which is what makes greedy good here.
    Candidates that do not fit are skipped rather than terminating the fill -- a small high-value
    weight further down the ranking should still get the room a huge one could not use.
    """
    budget_bytes = max(0, int(budget_bytes or 0))
    pinned = []
    skipped = []
    by_class = {}
    used = 0
    saved = 0.0

    for candidate in rank(candidates):
        if candidate.packed_bytes <= 0:
            skipped.append((candidate.key, "no packed bytes to keep resident"))
            continue
        if used + candidate.packed_bytes > budget_bytes:
            skipped.append((candidate.key,
                            f"needs {candidate.packed_bytes} bytes, {budget_bytes - used} left"))
            continue
        pinned.append(candidate.key)
        by_class[candidate.priority] = by_class.get(candidate.priority, 0) + 1
        used += candidate.packed_bytes
        saved += candidate.bytes_saved_per_token

    return PinPlan(pinned=tuple(pinned), bytes_pinned=used, budget_bytes=budget_bytes,
                   skipped=tuple(skipped), bytes_saved_per_token=saved, by_class=by_class)


def pin_budget_from(device_budget_bytes, window_budget_bytes):
    """What is left for pinning once the prefetch window has been accounted for.

    The window is not optional: a layer that is not resident still has to land somewhere before it
    can run, so its room is committed before anything is pinned. On a small card this subtraction
    is what legitimately produces a zero pin budget, and zero is a supported answer.
    """
    return max(0, int(device_budget_bytes or 0) - int(window_budget_bytes or 0))


class PinPlanner:
    """Keeps a plan current as the device budget moves, without reshuffling over noise.

    The budget already publishes with hysteresis, so what arrives here is a real change. It is still
    damped again, because the two questions are different: the budget's threshold asks whether the
    machine changed, and this one asks whether the change is worth the eviction and refetch that
    acting on it costs. A KV cache growing steadily through a long generation will trip the first
    repeatedly and should trip the second rarely.
    """

    def __init__(self, profile=None, replan_bytes=None, window_budget_bytes=None):
        self.profile = profile
        self.replan_bytes = self._knob("pin_replan_bytes", replan_bytes, 0)
        self.window_budget_bytes = self._knob("window_budget_bytes", window_budget_bytes, 0)
        self.plan = None
        self._planned_for = None
        self._candidates = ()
        self.replans = 0
        self.suppressed = 0

    def _knob(self, name, override, fallback):
        if override is not None:
            return int(override)
        if self.profile is not None:
            derivation = self.profile.derived.get(name)
            if derivation is not None:
                return int(derivation.value)
        return fallback

    def build(self, candidates, device_budget_bytes):
        """Plan from scratch for a device budget, ignoring damping. Use at load."""
        self._candidates = tuple(candidates)
        budget = pin_budget_from(device_budget_bytes, self.window_budget_bytes)
        self.plan = plan_pins(self._candidates, budget)
        self._planned_for = int(device_budget_bytes or 0)
        return self.plan

    def should_replan(self, device_budget_bytes):
        if self.plan is None:
            return True
        return abs(int(device_budget_bytes or 0) - self._planned_for) >= self.replan_bytes

    def budget_changed(self, old_bytes, new_bytes, sample=None):
        """Callback shape for :class:`~rocketllm.memory.budget.VramBudget`.

        Returns the new plan when one was built, and ``None`` when the move was damped away, so a
        caller can tell "nothing to do" from "here is a different set of pins".
        """
        if not self.should_replan(new_bytes):
            self.suppressed += 1
            return None
        previous = self.plan.pinned_set if self.plan is not None else frozenset()
        self.build(self._candidates, new_bytes)
        self._planned_for = int(new_bytes or 0)
        if self.plan.pinned_set == previous:
            # The budget moved enough to look at, and the answer came out the same. Nothing is
            # evicted for that.
            return None
        self.replans += 1
        log.debug("pin plan rebuilt for a %d -> %d byte budget: %s",
                  old_bytes or 0, new_bytes or 0, self.plan.explain())
        return self.plan

    def stats(self):
        return {
            "replans": self.replans,
            "suppressed": self.suppressed,
            "replan_threshold_bytes": self.replan_bytes,
            "window_budget_bytes": self.window_budget_bytes,
            "plan": self.plan.to_dict() if self.plan is not None else None,
        }
