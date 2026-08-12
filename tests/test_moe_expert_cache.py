"""Tests for hot-expert residency, routing statistics and intra-layer parallel fetch.

The residency policy here is a bet: that a mixture reads some experts far more than others, so
keeping the popular ones costs a fixed amount of memory and saves a growing amount of storage
traffic. Everything below pins some part of that bet.

Two properties matter more than the rest and are easy to lose in a refactor.

The first is *where the popularity count comes from*. It is the router's selection, never the set of
expert modules the forward happened to call. Several transformers releases run every expert in a
layer and multiply the unrouted ones by a zero weight, so counting calls reports a perfectly flat
distribution for a model that is in fact strongly skewed -- and a pin plan built from that is worse
than no plan, because it spends the budget on experts chosen at random.

The second is that the top-k reads are issued *together*. Serial reads of k experts cost k round
trips, each one latency-bound; issued at once they overlap. The test for that asserts real
concurrency with a barrier rather than counting calls, because a loop that submits k futures and
immediately waits on each in turn would pass any weaker check while behaving exactly like the serial
version it replaced.

Nothing here needs an accelerator or a checkpoint.
"""
import threading
import unittest

from rocketllm.memory import CLASS_ROUTED, CLASS_SHARED, TieredWeightCache, expert_kind
from rocketllm.memory.placement import plan_pins
from rocketllm.moe.expert_cache import (ExpertResidency, _gini, is_shared,
                                        shared_kind)

MB = 1024 * 1024


def expert_key(layer, index):
    return (layer, expert_kind(index))


def residency(layers=1, experts=8, expert_bytes=MB, cache=None, **kwargs):
    unit = ExpertResidency(cache=cache, **kwargs)
    for layer in range(layers):
        for index in range(experts):
            unit.track_expert(expert_key(layer, index), layer, index, expert_bytes)
    return unit


class TestRoutingStatistics(unittest.TestCase):
    def test_frequency_is_selections_per_visit_to_that_layer(self):
        unit = residency(experts=4)
        for _ in range(10):
            unit.stats.observe(0, (1, 2))
        self.assertAlmostEqual(unit.stats.frequency(expert_key(0, 1)), 1.0)
        self.assertAlmostEqual(unit.stats.frequency(expert_key(0, 2)), 1.0)
        self.assertEqual(unit.stats.frequency(expert_key(0, 3)), 0.0)

    def test_a_layer_is_rated_against_its_own_visits(self):
        """A mixture on every other layer must not be scored against layers it never saw."""
        unit = residency(layers=2, experts=4)
        for _ in range(10):
            unit.stats.observe(0, (0,))
        unit.stats.observe(1, (0,))
        self.assertAlmostEqual(unit.stats.frequency(expert_key(0, 0)), 1.0)
        self.assertAlmostEqual(unit.stats.frequency(expert_key(1, 0)), 1.0)

    def test_counts_come_from_the_router_not_from_which_modules_ran(self):
        """The distinction the whole ranking rests on.

        A forward that calls every expert and masks the unrouted ones must not flatten the counts.
        Only what the router chose is recorded, so the skew survives.
        """
        unit = residency(experts=4)
        for _ in range(20):
            unit.stats.observe(0, (0,))  # the router picked one; a forward may still run all four
        distribution = unit.stats.distribution()
        self.assertEqual(distribution["touched"], 1)
        self.assertAlmostEqual(distribution["top_1_share"], 1.0)
        self.assertGreater(distribution["gini"], 0.5)

    def test_aging_halves_the_counts(self):
        unit = residency(experts=4, aging_interval=4)
        for _ in range(4):
            unit.stats.observe(0, (0,))
        self.assertEqual(unit.stats.agings, 1)
        # Four selections, halved once: the ranking keeps its order at half the magnitude.
        self.assertAlmostEqual(unit.stats.frequency(expert_key(0, 0)), 0.5)

    def test_aging_lets_a_new_favourite_overtake_an_old_one(self):
        """Without this the first hundred tokens would decide residency for the whole run."""
        unit = residency(experts=4, aging_interval=8)
        for _ in range(8):
            unit.stats.observe(0, (0,))
        for _ in range(8):
            unit.stats.observe(0, (1,))
        self.assertGreater(unit.stats.frequency(expert_key(0, 1)),
                           unit.stats.frequency(expert_key(0, 0)))

    def test_raw_selections_survive_aging_for_reporting(self):
        """The printed distribution describes the run, not what the last halving left behind."""
        unit = residency(experts=4, aging_interval=2)
        for _ in range(10):
            unit.stats.observe(0, (0,))
        self.assertEqual(unit.stats.distribution()["selections"], 10)

    def test_nothing_observed_is_not_a_plan(self):
        unit = residency()
        self.assertFalse(unit.stats.observed)
        self.assertEqual([c for c in unit.pin_candidates() if c.priority == CLASS_ROUTED], [])

    def test_an_unknown_expert_is_ignored_rather_than_invented(self):
        unit = residency(experts=2)
        unit.stats.observe(0, (0, 99))
        self.assertEqual(unit.stats.distribution()["selections"], 1)


class TestSkewMeasurement(unittest.TestCase):
    """The distribution is printed because the policy depends on it. It has to be honest."""

    def test_uniform_routing_reports_no_skew(self):
        self.assertAlmostEqual(_gini([5, 5, 5, 5]), 0.0, places=6)

    def test_total_concentration_approaches_one(self):
        self.assertGreater(_gini([100] + [0] * 99), 0.95)

    def test_nothing_routed_is_zero_not_an_error(self):
        self.assertEqual(_gini([]), 0.0)
        self.assertEqual(_gini([0, 0]), 0.0)

    def test_the_uniform_share_is_reported_beside_the_measured_one(self):
        """So a flat distribution is visible rather than something a reader has to work out."""
        unit = residency(experts=10)
        for index in range(10):
            unit.stats.observe(0, (index,))
        distribution = unit.stats.distribution()
        self.assertAlmostEqual(distribution["top_10pct_share"], distribution["uniform_share"])

    def test_distinct_experts_per_visit(self):
        unit = residency(experts=8)
        for _ in range(5):
            unit.stats.observe(0, (0, 1, 2))
        self.assertAlmostEqual(unit.stats.distribution()["distinct_per_visit"], 3.0)


class TestPinCandidates(unittest.TestCase):
    def test_shared_experts_outrank_every_routed_expert(self):
        """A class boundary, not a score. No popularity promotes an expert past an always-on one."""
        unit = residency(experts=4)
        unit.track_shared((0, shared_kind("mlp.shared_expert")), 4 * MB)
        for _ in range(50):
            unit.stats.observe(0, (0,))

        candidates = {c.key: c for c in unit.pin_candidates()}
        shared = candidates[(0, shared_kind("mlp.shared_expert"))]
        hottest = candidates[expert_key(0, 0)]
        self.assertEqual(shared.priority, CLASS_SHARED)
        self.assertEqual(hottest.priority, CLASS_ROUTED)
        self.assertLess(shared.priority, hottest.priority)

    def test_a_shared_expert_is_pinned_before_a_bigger_budget_runs_out(self):
        unit = residency(experts=4, expert_bytes=MB)
        unit.track_shared((0, shared_kind("mlp.shared_expert")), 2 * MB)
        for _ in range(50):
            unit.stats.observe(0, (0,))

        plan = plan_pins(unit.pin_candidates(), 3 * MB)
        self.assertIn((0, shared_kind("mlp.shared_expert")), plan.pinned)
        self.assertIn(expert_key(0, 0), plan.pinned)

    def test_hot_experts_are_pinned_and_cold_ones_are_not(self):
        unit = residency(experts=8, expert_bytes=MB)
        for _ in range(100):
            unit.stats.observe(0, (0, 1))

        plan = plan_pins(unit.pin_candidates(), 2 * MB)
        self.assertEqual(set(plan.pinned), {expert_key(0, 0), expert_key(0, 1)})

    def test_an_expert_never_routed_to_is_offered_at_zero(self):
        unit = residency(experts=4)
        unit.stats.observe(0, (0,))
        cold = next(c for c in unit.pin_candidates() if c.key == expert_key(0, 3))
        self.assertEqual(cold.accesses_per_token, 0.0)
        self.assertEqual(cold.savings_per_resident_byte, 0.0)

    def test_shared_kinds_are_not_expert_kinds(self):
        """Filed as an expert they would get LFU, and win it trivially every time."""
        key = shared_kind("mlp.shared_expert")
        self.assertTrue(is_shared(key))
        self.assertFalse(key.startswith("expert:"))


class FakeCache:
    def __init__(self):
        self.prefetched = []
        self.pinned = set()

    def prefetch(self, keys):
        keys = list(keys)
        self.prefetched.append(tuple(keys))
        return len(keys)

    def report(self):
        return {}


class TestRouterDrivenFetch(unittest.TestCase):
    def test_the_whole_top_k_is_issued_on_one_call(self):
        """One call, all k keys: the point is that they go to the workers together."""
        cache = FakeCache()
        unit = residency(experts=8, cache=cache)
        unit.on_router(0, (2, 5, 7))
        self.assertEqual(cache.prefetched, [(expert_key(0, 2), expert_key(0, 5), expert_key(0, 7))])

    def test_fetching_happens_before_the_bookkeeping(self):
        """Every microsecond before the read is issued is one the drive spent idle."""
        order = []
        cache = FakeCache()
        cache.prefetch = lambda keys: (order.append("fetch"), len(list(keys)))[1]
        unit = residency(experts=4, cache=cache)
        original = unit.stats.observe
        unit.stats.observe = lambda *a, **k: (order.append("count"), original(*a, **k))[1]
        unit.on_router(0, (1,))
        self.assertEqual(order, ["fetch", "count"])

    def test_an_empty_selection_does_nothing(self):
        cache = FakeCache()
        unit = residency(cache=cache)
        self.assertEqual(unit.on_router(0, ()), 0)
        self.assertEqual(cache.prefetched, [])

    def test_no_cross_layer_lookahead_is_attempted(self):
        """Layer L+1's routing does not exist until L finishes; nothing may prefetch it."""
        cache = FakeCache()
        unit = residency(layers=3, experts=4, cache=cache)
        unit.on_router(1, (0, 2))
        touched_layers = {key[0] for batch in cache.prefetched for key in batch}
        self.assertEqual(touched_layers, {1})


class TestReplanCadence(unittest.TestCase):
    def test_the_first_rebuild_happens_as_soon_as_every_layer_has_routed(self):
        """The plan made at load holds no expert at all, so the first replacement cannot wait."""
        unit = residency(layers=3, experts=4, cache=FakeCache(), replan_interval=1000)
        calls = []
        unit.on_replan(lambda: calls.append(1) or "plan")

        unit.on_router(0, (0,))
        unit.on_router(1, (0,))
        self.assertEqual(calls, [], "replanned before the whole model had been seen")
        unit.on_router(2, (0,))
        self.assertEqual(len(calls), 1)

    def test_afterwards_it_settles_onto_the_interval(self):
        unit = residency(layers=1, experts=4, cache=FakeCache(), replan_interval=5)
        calls = []
        unit.on_replan(lambda: calls.append(1) or "plan")
        for _ in range(12):
            unit.on_router(0, (0,))
        # One warmup rebuild on the first firing, then one per interval.
        self.assertEqual(len(calls), 3)
        self.assertEqual(unit.replans, 3)

    def test_a_layer_that_never_routes_does_not_block_the_warmup_forever(self):
        """Only layers that actually hold experts are waited for."""
        unit = residency(layers=2, experts=4, cache=FakeCache(), replan_interval=1000)
        calls = []
        unit.on_replan(lambda: calls.append(1) or "plan")
        unit.on_router(0, (0,))
        unit.on_router(1, (1,))
        self.assertEqual(len(calls), 1)

    def test_no_hook_means_no_replan_and_no_error(self):
        unit = residency(experts=4, cache=FakeCache(), replan_interval=1)
        for _ in range(3):
            unit.on_router(0, (0,))
        self.assertEqual(unit.replans, 0)


class TestCachePrefetchesInParallel(unittest.TestCase):
    """The parallel half of the win, asserted by real concurrency rather than by call counting."""

    def test_named_keys_are_read_concurrently(self):
        width = 4
        barrier = threading.Barrier(width, timeout=10)

        def fetch(key):
            # Every read must be in flight at once for this to return; a serial implementation
            # deadlocks here and the barrier times out.
            barrier.wait()
            return f"payload:{key}"

        cache = TieredWeightCache(fetch=fetch, sizer=lambda key: MB, device_bytes=64 * MB,
                                  sequence=lambda key, w: [], prefetch_workers=width)
        try:
            keys = [expert_key(0, i) for i in range(width)]
            self.assertEqual(cache.prefetch(keys), width)
            for key in keys:
                self.assertEqual(cache.acquire(key), f"payload:{key}")
                cache.release(key)
        finally:
            cache.close()

    def test_a_key_already_in_flight_is_not_read_twice(self):
        reads = []
        cache = TieredWeightCache(fetch=lambda key: reads.append(key) or f"payload:{key}",
                                  sizer=lambda key: MB, device_bytes=64 * MB,
                                  sequence=lambda key, w: [], prefetch_workers=2)
        try:
            key = expert_key(0, 0)
            cache.prefetch([key])
            cache.prefetch([key])
            cache.acquire(key)
            cache.release(key)
            self.assertEqual(reads, [key])
        finally:
            cache.close()

    def test_a_resident_key_is_not_prefetched(self):
        reads = []
        cache = TieredWeightCache(fetch=lambda key: reads.append(key) or f"payload:{key}",
                                  sizer=lambda key: MB, device_bytes=64 * MB,
                                  sequence=lambda key, w: [], prefetch_workers=2)
        try:
            key = expert_key(0, 0)
            cache.acquire(key)
            cache.release(key)
            self.assertEqual(cache.prefetch([key]), 0)
            self.assertEqual(reads, [key])
        finally:
            cache.close()

    def test_a_prefetched_read_is_collected_rather_than_repeated(self):
        reads = []
        cache = TieredWeightCache(fetch=lambda key: reads.append(key) or f"payload:{key}",
                                  sizer=lambda key: MB, device_bytes=64 * MB,
                                  sequence=lambda key, w: [], prefetch_workers=2)
        try:
            key = expert_key(0, 3)
            cache.prefetch([key])
            cache.acquire(key)
            cache.release(key)
            self.assertEqual(len(reads), 1)
            self.assertEqual(cache.report()["prefetch_hits"], 1)
        finally:
            cache.close()

    def test_prefetching_disabled_is_not_an_error(self):
        """No worker pool means the reads happen when each expert runs, which is still correct."""
        cache = TieredWeightCache(fetch=lambda key: "payload", sizer=lambda key: MB,
                                  device_bytes=64 * MB, prefetch_workers=0)
        try:
            self.assertEqual(cache.prefetch([expert_key(0, 0)]), 0)
        finally:
            cache.close()


class TestExpertScopedCounters(unittest.TestCase):
    """A mixture's dense and expert entries need separate hit rates; one average hides the answer."""

    def test_expert_hits_are_counted_apart_from_dense_ones(self):
        cache = TieredWeightCache(fetch=lambda key: "payload", sizer=lambda key: MB,
                                  device_bytes=64 * MB, window=8)
        try:
            for key in (expert_key(0, 1), (0, "dense")):
                cache.acquire(key)
                cache.release(key)
                cache.acquire(key)
                cache.release(key)
            report = cache.report()
            self.assertEqual(report["expert_misses"], 1)
            self.assertEqual(report["expert_hits_device"], 1)
            self.assertEqual(report["misses"], 2)
            self.assertEqual(report["hits_device"], 2)
        finally:
            cache.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
