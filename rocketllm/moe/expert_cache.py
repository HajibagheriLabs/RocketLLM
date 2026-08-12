"""Keeping the experts a mixture actually uses, and fetching the rest as a batch.

Three things live here, and they exist because a mixture behaves nothing like a dense model under a
cache.

**Popularity, measured rather than assumed.** Decoder layers are read cyclically, so no recency or
frequency rule helps: every layer is read exactly once per token and the scan is longer than the
cache. Experts are the opposite. A token visits a handful of the hundreds in a layer, the same
handful keeps coming up, and a count therefore predicts the next access well -- which is why the
cache runs LFU-with-aging for experts and FIFO-plus-pinning for everything else. That asymmetry is
only worth anything if the skew is real, so this module measures it online instead of asserting it,
and the benchmark prints the distribution. If a model turns out to route uniformly, the numbers will
say so.

Frequency is counted from what the ROUTER selected, never from which expert modules the forward
happened to call. Those are not the same thing: several transformers releases walk every expert in
the layer and multiply the unrouted ones by a zero weight, so counting calls would report a
perfectly flat distribution for a model that is in fact strongly skewed, and the pin plan built from
it would be worthless.

**Aging.** A raw count only ever grows, so an expert that was hot for the first two hundred tokens
would outrank one that is hot now for the remainder of a long generation. Halving every count on a
fixed interval keeps the ordering while letting recent behaviour overtake old behaviour, which is
what makes this adapt when the topic shifts mid-conversation.

**Fetching the top-k together.** The router names every expert the layer is about to use, all at
once. Read serially, k experts cost k round trips to storage, each one latency-bound and none of
them reaching the drive's rated bandwidth. Issued together they overlap, which is the entire reason
the loader has a measured worker count. This is the only lookahead a mixture permits: layer L's
router runs inside layer L, so layer L+1's experts are not merely unknown but undefined, and nothing
here may be built as if they were.
"""
import dataclasses
import logging
import threading

from ..memory import CLASS_ROUTED, CLASS_SHARED, PinCandidate, expert_kind

log = logging.getLogger(__name__)

#: Kind marker for an always-on module streamed beside the routed experts. Deliberately not an
#: expert kind: these are read on every token, so the cache must give them the dense policy and not
#: rank them against experts by popularity they trivially win.
SHARED_PREFIX = "shared:"


def shared_kind(path):
    return f"{SHARED_PREFIX}{path}"


def is_shared(kind):
    return str(kind).startswith(SHARED_PREFIX)


@dataclasses.dataclass
class ExpertRecord:
    """One routed expert, as the cache and the pin planner need to see it."""

    key: tuple
    layer: int
    index: int
    packed_bytes: int
    #: Aged selection count. Not a raw total -- see :meth:`ExpertStats.age`.
    uses: float = 0.0
    #: Raw lifetime selections, for reporting. Never aged, so the printed distribution describes the
    #: whole run rather than whatever survived the last halving.
    selections: int = 0


class ExpertStats:
    """Online routing frequency, with aging, per expert.

    Fed one router selection at a time. Everything it answers is per *router firing* -- one layer
    being visited once -- because that is the event the router actually produces. During decode a
    firing is one token; during prefill it covers the whole prompt at once, so the union of experts
    it reports is wider than any single token's.
    """

    # The literal is a floor for direct construction, not a tuning choice: the engine passes the
    # profile's `expert_aging_interval`, and this value only applies when something builds an
    # ExpertStats without a profile to hand (a test, or a caller reading the counts on their own).
    # It describes how fast popularity should decay, which is a property of the routing rather than
    # of any machine, so no hardware measurement feeds it.
    def __init__(self, aging_interval=4096):
        self.aging_interval = max(1, int(aging_interval))
        self._records = {}
        self._lock = threading.Lock()
        self.firings = 0
        self.selections = 0
        self.agings = 0
        #: Firings per layer, so a frequency is a share of the visits that layer actually had. A
        #: model that routes only every other layer would otherwise have its mixture layers rated
        #: against a denominator they never saw.
        self._layer_firings = {}

    # -- recording -------------------------------------------------------------------------------

    def track(self, record):
        """Register an expert before anything has routed to it."""
        self._records[record.key] = record

    def observe(self, layer, selected):
        """Record one router firing: the experts this layer just chose."""
        with self._lock:
            self.firings += 1
            self._layer_firings[layer] = self._layer_firings.get(layer, 0) + 1
            for index in selected:
                record = self._records.get((layer, expert_kind(index)))
                if record is None:
                    continue
                record.uses += 1.0
                record.selections += 1
                self.selections += 1
            if self.firings % self.aging_interval == 0:
                self._age_locked()

    def _age_locked(self):
        for record in self._records.values():
            record.uses *= 0.5
        self.agings += 1

    def age(self):
        with self._lock:
            self._age_locked()

    # -- what the planner asks --------------------------------------------------------------------

    def frequency(self, key):
        """Selections per visit to that expert's layer, in ``0..1``.

        This is `accesses_per_token` in the placement module's sense: how much storage traffic one
        resident copy saves per token. An expert chosen on every visit scores 1.0 and is worth as
        much resident as an attention block; one never chosen scores 0 and is worth nothing.
        """
        record = self._records.get(key)
        if record is None:
            return 0.0
        visits = self._layer_firings.get(record.layer, 0)
        if not visits:
            return 0.0
        return min(1.0, record.uses / visits)

    def records(self):
        return tuple(self._records.values())

    @property
    def observed(self):
        """Whether anything has routed yet. A plan built before this is guesswork."""
        return self.firings > 0

    @property
    def layers_seen(self):
        """Layers whose router has fired at least once."""
        return frozenset(self._layer_firings)

    @property
    def layers_tracked(self):
        """Layers holding experts, whether or not they have routed yet."""
        return frozenset(record.layer for record in self._records.values())

    # -- reporting ---------------------------------------------------------------------------------

    def distribution(self):
        """The shape of the skew the whole approach depends on.

        Reported from raw lifetime selections rather than the aged counts, because this describes
        what the run did and not what the ranking currently believes.
        """
        counts = sorted((r.selections for r in self._records.values()), reverse=True)
        total = sum(counts)
        touched = sum(1 for c in counts if c)
        summary = {
            "experts": len(counts),
            "touched": touched,
            "firings": self.firings,
            "selections": total,
            "agings": self.agings,
            "distinct_per_visit": (self.selections / self.firings) if self.firings else 0.0,
        }
        if not total:
            summary.update({"top_1_share": 0.0, "top_10pct_share": 0.0, "gini": 0.0,
                            "uniform_share": 0.0})
            return summary

        head = max(1, len(counts) // 10)
        summary["top_1_share"] = counts[0] / total
        summary["top_10pct_share"] = sum(counts[:head]) / total
        # What the top decile would hold if every expert were equally popular. Printed beside the
        # measured share so a flat distribution is obvious rather than something to be inferred.
        summary["uniform_share"] = head / len(counts) if counts else 0.0
        summary["gini"] = _gini(counts)
        return summary


def _gini(counts):
    """Inequality of a selection distribution: 0 is perfectly uniform, 1 is one expert taking all.

    A single number for "is the skew real", which is the question the residency policy rests on.
    """
    values = sorted(counts)
    n = len(values)
    total = sum(values)
    if n == 0 or total == 0:
        return 0.0
    weighted = sum((i + 1) * value for i, value in enumerate(values))
    return max(0.0, min(1.0, (2.0 * weighted) / (n * total) - (n + 1.0) / n))


class ExpertResidency:
    """Ties routing to the cache: what to prefetch now, and what to keep.

    Built once per model. The engine hands it every router firing; it answers by starting the reads
    for that layer's top-k and by folding the selection into the popularity counts. Periodically it
    rebuilds the pin plan from those counts and hands the result back to the cache.
    """

    # Both intervals are floors for direct construction; the engine supplies the profile's
    # `expert_aging_interval` and `expert_replan_interval`. Neither is a hardware quantity -- they
    # are timescales in units of router firings, and they mean the same thing on any device.
    def __init__(self, cache=None, aging_interval=4096, replan_interval=512, enabled=True):
        self.cache = cache
        self.stats = ExpertStats(aging_interval=aging_interval)
        self.replan_interval = max(1, int(replan_interval))
        self.enabled = enabled
        #: Always-on modules: key -> packed bytes. Never ranked by popularity, only by size within
        #: their class, because every token reads all of them.
        self.shared = {}
        self._since_replan = 0
        self._replan_hook = None
        self.replans = 0
        self.prefetched = 0
        self.prefetch_calls = 0

    # -- registration ------------------------------------------------------------------------------

    def track_expert(self, key, layer, index, packed_bytes):
        self.stats.track(ExpertRecord(key=key, layer=layer, index=index,
                                      packed_bytes=int(packed_bytes)))

    def track_shared(self, key, packed_bytes):
        self.shared[key] = int(packed_bytes)

    def on_replan(self, callback):
        """Called with a fresh pin plan whenever the ranking has moved enough to be worth acting on."""
        self._replan_hook = callback

    @property
    def experts(self):
        return self.stats.records()

    # -- the router firing --------------------------------------------------------------------------

    def on_router(self, layer, selected):
        """A layer's router has just chosen. Start those reads, and remember the choice.

        Order matters here: the fetches are issued before anything else is done with the selection,
        because the whole value of knowing it is the head start, and every microsecond spent first
        is a microsecond the drive was idle.
        """
        if not self.enabled or not selected:
            return 0
        started = self._prefetch(layer, selected)
        self.stats.observe(layer, selected)
        self._since_replan += 1
        if self._should_replan():
            self._since_replan = 0
            self.replan()
        return started

    def _should_replan(self):
        """Whether the ranking has moved enough to be worth acting on.

        Normally: once per interval, because rebuilding a plan evicts weights and re-reads them, and
        that has to be worth more than the residency it buys back.

        The first rebuild is the exception. The plan made at load was built before anything had
        routed, so it contains no expert at all -- waiting a full interval to replace it means a
        short generation never gets expert residency, and the first one is by far the most valuable.
        It therefore fires as soon as the plan can be evidence-based at all, which is when every
        layer holding experts has routed at least once.
        """
        if self._since_replan >= self.replan_interval:
            return True
        if self.replans:
            return False
        tracked = self.stats.layers_tracked
        return bool(tracked) and tracked <= self.stats.layers_seen

    def _prefetch(self, layer, selected):
        """Issue the layer's top-k reads together rather than one per expert as each runs."""
        if self.cache is None:
            return 0
        keys = [(layer, expert_kind(index)) for index in selected]
        started = self.cache.prefetch(keys)
        if started:
            self.prefetched += started
            self.prefetch_calls += 1
        return started

    # -- residency ----------------------------------------------------------------------------------

    def pin_candidates(self):
        """What a mixture offers the pin planner, in the planner's own priority classes.

        Shared modules first: every token reads them, so their value per resident byte does not
        depend on anything being measured, and no routed expert should displace one however popular
        it turns out to be. Routed experts follow, ranked by the frequency actually observed. An
        expert nothing has routed to yet is offered at zero and will simply lose.
        """
        if not self.enabled:
            # The override that turns this whole policy off: experts are still cached, they simply
            # never earn a pin, which is the behaviour before any of this existed. Kept so the
            # policy can be measured against its own absence on one binary rather than two.
            return []
        candidates = [PinCandidate(key=key, packed_bytes=size, priority=CLASS_SHARED,
                                   accesses_per_token=1.0, label="shared expert")
                      for key, size in self.shared.items()]
        if self.stats.observed:
            candidates.extend(
                PinCandidate(key=record.key, packed_bytes=record.packed_bytes,
                             priority=CLASS_ROUTED,
                             accesses_per_token=self.stats.frequency(record.key),
                             label=f"expert {record.index} of layer {record.layer}")
                for record in self.stats.records())
        return candidates

    def replan(self):
        """Rebuild the pin plan from the counts as they now stand."""
        if self._replan_hook is None:
            return None
        plan = self._replan_hook()
        if plan is not None:
            self.replans += 1
        return plan

    # -- reporting -------------------------------------------------------------------------------

    def report(self):
        """Everything the benchmark prints about the mixture."""
        summary = dict(self.stats.distribution())
        summary.update({
            "enabled": self.enabled,
            "shared_modules": len(self.shared),
            "shared_bytes": sum(self.shared.values()),
            "replans": self.replans,
            "prefetched": self.prefetched,
            "prefetch_calls": self.prefetch_calls,
            "replan_interval": self.replan_interval,
            "aging_interval": self.stats.aging_interval,
        })
        if self.cache is not None:
            cache_report = self.cache.report()
            hits = cache_report.get("expert_hits_device", 0) + cache_report.get("expert_hits_host", 0)
            total = hits + cache_report.get("expert_misses", 0)
            summary.update({
                "acquires": total,
                "hits_device": cache_report.get("expert_hits_device", 0),
                "hits_host": cache_report.get("expert_hits_host", 0),
                "misses": cache_report.get("expert_misses", 0),
                "hit_rate": (hits / total) if total else 0.0,
                "pinned_experts": sum(1 for key in self.cache.pinned
                                      if str(key[1]).startswith("expert:")),
                "pinned_shared": sum(1 for key in self.cache.pinned if is_shared(key[1])),
            })
        return summary
