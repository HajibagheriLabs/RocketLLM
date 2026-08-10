"""Policy tests for the tiered weight cache and the pin planner.

The cache is the component whose bugs do not raise. A wrong replacement policy still returns the
right bytes -- it just returns them after reading them from disk again, and the only symptom is that
the machine is slow in a way that looks like the machine's fault. So the policies are pinned here by
their observable consequences: what gets evicted, in what order, and what the hit rate comes out at
for an access pattern that matches how decode actually walks a model.

Every tier is mocked. There is no accelerator, no checkpoint and no I/O: "storage" is a dict and a
fetch counter, which is what makes it possible to assert on the exact number of storage reads.
"""
import unittest

from rocketllm.memory.cache import TieredWeightCache, expert_kind, is_expert
from rocketllm.memory.placement import (CLASS_ALWAYS, CLASS_ROUTED, CLASS_SHARED, PinCandidate,
                                        PinPlanner, pin_budget_from, plan_pins, rank)

MB = 1024 * 1024


def dense_key(layer):
    return (layer, "dense")


def expert_key(layer, expert):
    return (layer, expert_kind(expert))


class FakeStorage:
    """Stands in for the checkpoint. Counts reads, because that is the metric that matters."""

    def __init__(self, size_bytes=10 * MB):
        self.reads = []
        self.size_bytes = size_bytes
        self.sizes = {}

    def fetch(self, key):
        self.reads.append(key)
        return f"payload:{key}"

    def size(self, key):
        return self.sizes.get(key, self.size_bytes)

    @property
    def read_count(self):
        return len(self.reads)


def build_cache(layers=8, layer_bytes=10 * MB, device_bytes=None, host_bytes=0, pinned=(),
                window=2, **kwargs):
    storage = FakeStorage(layer_bytes)
    if device_bytes is None:
        device_bytes = layer_bytes * (len(pinned) + window)
    cache = TieredWeightCache(fetch=storage.fetch, sizer=storage.size, device_bytes=device_bytes,
                              host_bytes=host_bytes, pinned=pinned, window=window, **kwargs)
    return cache, storage


def walk(cache, keys):
    """Acquire and release each key in turn, the way a forward pass does."""
    for key in keys:
        cache.acquire(key)
        cache.release(key)


class TestCyclicAccessIsNotLru(unittest.TestCase):
    """REGRESSION TEST. This exists specifically to catch someone swapping the dense policy to LRU.

    Decode walks the decoder layers cyclically -- 0, 1, ... L-1, 0, 1, ... L-1 -- and when the cache
    holds fewer layers than the model has, LRU is exactly the wrong policy for that pattern. The
    least recently used layer is always the one the cycle is about to come round to next, so LRU
    evicts precisely the entry needed soonest, every time, and the hit rate collapses to zero.

    The cache defends against this with a statically pinned subset plus a FIFO prefetch window. What
    is asserted below is the consequence: over a cyclic scan larger than the cache, the hit rate
    equals the pinned fraction, and the pinned entries are read from storage exactly once no matter
    how many cycles run. If someone replaces the policy with LRU over the whole device pool, the
    pinned entries start being evicted, storage reads scale with the number of cycles, and these
    assertions fail.

    If you are changing this test to make a new policy pass, you are removing the only thing
    standing between this project and a ~0% cache hit rate on every model that does not fit.
    """

    LAYERS = 12
    PINNED = (dense_key(0), dense_key(1), dense_key(2))
    CYCLES = 10

    def run_scan(self):
        cache, storage = build_cache(layers=self.LAYERS, pinned=self.PINNED, window=2)
        keys = [dense_key(i) for i in range(self.LAYERS)]
        for _ in range(self.CYCLES):
            walk(cache, keys)
        return cache, storage, keys

    def test_the_hit_rate_equals_the_pinned_fraction(self):
        cache, _, _ = self.run_scan()
        report = cache.report()
        # Every access to a pinned layer hits except the first, which had to fetch it; no unpinned
        # layer can ever hit, because a scan longer than the cache never comes back to one while it
        # is still resident. So the count is exact, not approximate, and asserting it exactly is
        # what makes this a tripwire rather than a smoke test.
        hits = len(self.PINNED) * (self.CYCLES - 1)
        total = self.LAYERS * self.CYCLES
        self.assertEqual(report["hits_device"] + report["hits_host"], hits,
                         "the pinned subset is not hitting on every cycle -- is the dense policy "
                         "evicting pinned entries, i.e. has it become LRU over the whole pool?")
        self.assertAlmostEqual(report["hit_rate"], hits / total, places=9)

    def test_the_hit_rate_converges_on_the_pinned_fraction(self):
        """Asymptotically the cold start washes out and the pinned fraction is all that is left."""
        cache, _ = build_cache(layers=self.LAYERS, pinned=self.PINNED, window=2)
        keys = [dense_key(i) for i in range(self.LAYERS)]
        for _ in range(200):
            walk(cache, keys)
        self.assertAlmostEqual(cache.report()["hit_rate"], len(self.PINNED) / self.LAYERS,
                               delta=0.005)

    def test_pinned_layers_are_read_from_storage_exactly_once(self):
        _, storage, _ = self.run_scan()
        for key in self.PINNED:
            self.assertEqual(storage.reads.count(key), 1,
                             f"{key} is pinned but was re-read from storage; pinned entries must "
                             f"never be evicted")

    def test_storage_reads_do_not_grow_with_the_pinned_subset(self):
        """Under LRU every layer is re-read every cycle. Under this policy the pinned ones are not."""
        _, storage, _ = self.run_scan()
        unpinned = self.LAYERS - len(self.PINNED)
        # Each cycle re-reads every unpinned layer; the pinned ones are read once, at the start.
        expected = unpinned * self.CYCLES + len(self.PINNED)
        self.assertEqual(storage.read_count, expected)

    def test_a_longer_run_does_not_degrade_the_hit_rate(self):
        """LRU's collapse is total, so a regression would show as a hit rate that stays near zero."""
        short = build_cache(pinned=self.PINNED, window=2)
        keys = [dense_key(i) for i in range(self.LAYERS)]
        walk(short[0], keys * 2)
        long = build_cache(pinned=self.PINNED, window=2)
        walk(long[0], keys * 20)
        self.assertGreater(long[0].report()["hit_rate"], short[0].report()["hit_rate"] - 0.02)
        self.assertGreater(long[0].report()["hit_rate"], 0.2)


class TestFifoIsNotLru(unittest.TestCase):
    """Directly discriminating: an access pattern where FIFO and LRU choose different victims."""

    def test_the_oldest_admitted_is_evicted_not_the_least_recently_used(self):
        cache, storage = build_cache(window=3, device_bytes=3 * 10 * MB)
        walk(cache, [dense_key(0), dense_key(1), dense_key(2)])

        # Re-touch the oldest. Under LRU this makes layer 0 the most recently used and layer 1 the
        # next victim; under FIFO layer 0 is still the oldest admitted and goes first.
        cache.acquire(dense_key(0))
        cache.release(dense_key(0))

        walk(cache, [dense_key(3)])

        self.assertEqual(cache.tier_of(dense_key(0)), "storage",
                         "layer 0 survived: the policy is honouring recency, i.e. it is LRU")
        self.assertEqual(cache.tier_of(dense_key(1)), "device")

    def test_a_repeated_hit_does_not_extend_an_entrys_life(self):
        cache, _ = build_cache(window=2, device_bytes=2 * 10 * MB)
        walk(cache, [dense_key(0), dense_key(1)])
        for _ in range(5):
            walk(cache, [dense_key(0)])
        walk(cache, [dense_key(2)])
        self.assertEqual(cache.tier_of(dense_key(0)), "storage")


class TestZeroBudget(unittest.TestCase):
    """A device with no room to pin anything is a real user, not an error case."""

    def test_the_cache_degrades_to_pure_streaming_and_still_returns_entries(self):
        cache, storage = build_cache(device_bytes=0, host_bytes=0, pinned=(), window=1)
        keys = [dense_key(i) for i in range(4)]
        for _ in range(3):
            for key in keys:
                self.assertEqual(cache.acquire(key), f"payload:{key}")
                cache.release(key)
        self.assertEqual(cache.report()["hit_rate"], 0.0)
        self.assertEqual(storage.read_count, 12, "pure streaming reads every entry every pass")

    def test_nothing_is_retained_and_nothing_raises(self):
        cache, _ = build_cache(device_bytes=0, host_bytes=0)
        walk(cache, [dense_key(0), expert_key(0, 3)])
        self.assertEqual(cache.report()["device_entries"], 0)
        self.assertEqual(cache.report()["host_entries"], 0)

    def test_a_zero_budget_pin_plan_is_empty_rather_than_an_error(self):
        candidates = [PinCandidate(dense_key(i), 10 * MB) for i in range(4)]
        plan = plan_pins(candidates, 0)
        self.assertEqual(plan.pinned, ())
        self.assertTrue(plan.is_pure_streaming)
        self.assertIn("nothing is pinned", plan.explain())

    def test_an_entry_larger_than_the_whole_device_still_runs(self):
        """Not even one layer fitting is the engine's error to raise, not a crash in here."""
        cache, storage = build_cache(device_bytes=1 * MB, layer_bytes=10 * MB)
        self.assertEqual(cache.acquire(dense_key(0)), f"payload:{dense_key(0)}")
        cache.release(dense_key(0))
        self.assertEqual(cache.report()["rejected_too_large"], 1)

    def test_a_streamed_entry_is_handed_back_when_it_is_released(self):
        """REGRESSION TEST. Pure streaming has to actually release what it streamed.

        An entry admitted with nowhere to keep it belongs to no tier, so nothing else in the cache
        can ever discard it -- eviction only walks the tiers. If release does not hand it back, the
        owner is never told to unbind those weights and they stay on the device for the rest of the
        run. The cache reports an empty device tier throughout, so the symptom is a model that
        quietly materialises itself while every counter says it is streaming.
        """
        discarded = []
        cache, _ = build_cache(device_bytes=0, host_bytes=0, window=1,
                               discard=discarded.append)
        for key in (dense_key(0), expert_key(0, 3)):
            held = len(discarded)
            cache.acquire(key)
            self.assertEqual(len(discarded), held, "released before the reader was done with it")
            cache.release(key)
            self.assertEqual(discarded[-1], f"payload:{key}")
        self.assertEqual(cache.report()["entries"], 0, "a released transient entry is still tracked")

    def test_a_streamed_entry_survives_a_nested_claim(self):
        """Two readers, one payload: it goes back only when the last of them is finished."""
        discarded = []
        cache, storage = build_cache(device_bytes=0, host_bytes=0, discard=discarded.append)
        key = dense_key(0)
        cache.acquire(key)
        cache.acquire(key)
        self.assertEqual(storage.read_count, 1, "the second claim re-read a payload already held")
        cache.release(key)
        self.assertEqual(discarded, [])
        cache.release(key)
        self.assertEqual(discarded, [f"payload:{key}"])


class TestFitsEntirely(unittest.TestCase):
    """A device that holds the whole model must never evict anything."""

    def test_nothing_is_ever_evicted_when_everything_fits(self):
        layers = 8
        cache, storage = build_cache(device_bytes=layers * 10 * MB, window=layers)
        keys = [dense_key(i) for i in range(layers)]
        for _ in range(5):
            walk(cache, keys)
        report = cache.report()
        self.assertEqual(report["evicted_to_host"], 0)
        self.assertEqual(report["evicted_to_storage"], 0)
        self.assertEqual(storage.read_count, layers, "every layer read exactly once, then resident")

    def test_the_hit_rate_reaches_one_after_the_first_pass(self):
        layers = 6
        cache, _ = build_cache(device_bytes=layers * 10 * MB, window=layers)
        keys = [dense_key(i) for i in range(layers)]
        walk(cache, keys)
        walk(cache, keys * 9)
        report = cache.report()
        self.assertGreater(report["hit_rate"], 0.85)

    def test_a_budget_covering_everything_pins_everything(self):
        candidates = [PinCandidate(dense_key(i), 10 * MB) for i in range(6)]
        plan = plan_pins(candidates, 6 * 10 * MB)
        self.assertEqual(len(plan.pinned), 6)
        self.assertEqual(plan.skipped, ())


class TestRefcountingAndPinning(unittest.TestCase):
    def test_an_acquired_entry_is_never_evicted_underneath_its_reader(self):
        cache, _ = build_cache(window=1, device_bytes=1 * 10 * MB)
        cache.acquire(dense_key(0))
        walk(cache, [dense_key(1), dense_key(2), dense_key(3)])
        self.assertEqual(cache.tier_of(dense_key(0)), "device",
                         "an in-use entry was evicted mid-forward")
        cache.release(dense_key(0))

    def test_an_entry_becomes_evictable_once_released(self):
        cache, _ = build_cache(window=1, device_bytes=1 * 10 * MB)
        cache.acquire(dense_key(0))
        cache.release(dense_key(0))
        walk(cache, [dense_key(1), dense_key(2)])
        self.assertEqual(cache.tier_of(dense_key(0)), "storage")

    def test_nested_acquires_need_matching_releases(self):
        cache, _ = build_cache(window=1, device_bytes=1 * 10 * MB)
        cache.acquire(dense_key(0))
        cache.acquire(dense_key(0))
        cache.release(dense_key(0))
        walk(cache, [dense_key(1), dense_key(2)])
        self.assertEqual(cache.tier_of(dense_key(0)), "device")
        cache.release(dense_key(0))
        walk(cache, [dense_key(3), dense_key(4)])
        self.assertEqual(cache.tier_of(dense_key(0)), "storage")

    def test_pinned_entries_survive_pressure_that_evicts_everything_else(self):
        pinned = (dense_key(0),)
        cache, _ = build_cache(pinned=pinned, window=1, device_bytes=2 * 10 * MB)
        walk(cache, [dense_key(0)])
        walk(cache, [dense_key(i) for i in range(1, 10)])
        self.assertEqual(cache.tier_of(dense_key(0)), "device")

    def test_a_new_plan_releases_what_it_no_longer_pins(self):
        cache, _ = build_cache(pinned=(dense_key(0),), window=1, device_bytes=2 * 10 * MB)
        walk(cache, [dense_key(0)])
        cache.apply_plan([dense_key(5)])
        walk(cache, [dense_key(i) for i in range(1, 8)])
        self.assertEqual(cache.tier_of(dense_key(0)), "storage")


class TestHostTier(unittest.TestCase):
    def test_eviction_goes_to_host_before_storage(self):
        cache, _ = build_cache(window=1, device_bytes=1 * 10 * MB, host_bytes=4 * 10 * MB)
        walk(cache, [dense_key(0), dense_key(1)])
        self.assertEqual(cache.tier_of(dense_key(0)), "host")
        self.assertEqual(cache.report()["evicted_to_host"], 1)
        self.assertEqual(cache.report()["evicted_to_storage"], 0)

    def test_a_host_hit_avoids_a_storage_read(self):
        cache, storage = build_cache(window=1, device_bytes=1 * 10 * MB, host_bytes=4 * 10 * MB)
        walk(cache, [dense_key(0), dense_key(1)])
        before = storage.read_count
        walk(cache, [dense_key(0)])
        self.assertEqual(storage.read_count, before, "a host hit still went to storage")
        self.assertEqual(cache.report()["hits_host"], 1)

    def test_a_near_zero_host_tier_falls_straight_through_to_storage(self):
        """Little free RAM is common and must degrade, not fail."""
        cache, _ = build_cache(window=1, device_bytes=1 * 10 * MB, host_bytes=0)
        walk(cache, [dense_key(0), dense_key(1)])
        self.assertEqual(cache.tier_of(dense_key(0)), "storage")
        self.assertEqual(cache.report()["evicted_to_host"], 0)

    def test_a_full_host_tier_drops_its_coldest_rather_than_refusing(self):
        cache, _ = build_cache(window=1, device_bytes=1 * 10 * MB, host_bytes=2 * 10 * MB)
        walk(cache, [dense_key(i) for i in range(5)])
        report = cache.report()
        self.assertLessEqual(report["host_bytes"], report["host_capacity"])
        self.assertGreater(report["host_evictions"], 0)


class TestExpertPolicy(unittest.TestCase):
    """Experts use LFU with aging. The asymmetry with dense FIFO is deliberate."""

    def test_expert_granularity_caches_what_a_whole_layer_cannot(self):
        """Why a mixture is worth streaming per expert even when every expert gets read.

        A device budget smaller than one layer cannot hold that layer at all: the entry does not fit
        the tier, so it is rejected outright and asking for it a thousand times still reads it a
        thousand times. The same budget holds a useful number of that layer's experts. Granularity
        is what turns "nothing fits" into "most of it fits", and it does that whether or not the
        model's forward skips the experts a token did not route to -- which some do and some, at any
        given transformers version, do not.
        """
        budget = 50 * MB

        whole, whole_storage = build_cache(device_bytes=budget, layer_bytes=100 * MB, window=1)
        for _ in range(3):
            walk(whole, [dense_key(0)])
        self.assertEqual(whole.report()["hit_rate"], 0.0)
        self.assertEqual(whole_storage.read_count, 3, "a layer too large to cache is re-read")

        split, split_storage = build_cache(device_bytes=budget, layer_bytes=10 * MB, window=1)
        experts = [expert_key(0, e) for e in range(4)]
        for _ in range(3):
            walk(split, experts)
        self.assertEqual(split_storage.read_count, 4, "each expert should be read once, then hit")
        self.assertAlmostEqual(split.report()["hit_rate"], 8 / 12, places=6)

    def test_the_least_popular_expert_is_evicted_first(self):
        cache, _ = build_cache(device_bytes=3 * 10 * MB, window=1)
        walk(cache, [expert_key(0, 1)] * 5)
        walk(cache, [expert_key(0, 2)] * 1)
        walk(cache, [expert_key(0, 3)] * 3)
        walk(cache, [expert_key(0, 4)])
        self.assertEqual(cache.tier_of(expert_key(0, 2)), "storage",
                         "the coldest expert should have gone first")
        self.assertEqual(cache.tier_of(expert_key(0, 1)), "device")

    def test_experts_are_evicted_before_dense_layers(self):
        """A dense layer will be back within one cycle; a cold expert may not be for thousands."""
        cache, _ = build_cache(device_bytes=2 * 10 * MB, window=2)
        walk(cache, [dense_key(0), expert_key(0, 7)])
        walk(cache, [dense_key(1)])
        self.assertEqual(cache.tier_of(expert_key(0, 7)), "storage")
        self.assertEqual(cache.tier_of(dense_key(0)), "device")

    def test_aging_lets_a_newly_hot_expert_overtake_an_old_favourite(self):
        cache, _ = build_cache(device_bytes=8 * 10 * MB, window=1, aging_interval=4)
        walk(cache, [expert_key(0, 1)] * 8)
        before = cache._entries[expert_key(0, 1)].uses
        walk(cache, [expert_key(0, 2)] * 8)
        after = cache._entries[expert_key(0, 1)].uses
        self.assertLess(after, before, "old popularity never decayed; LFU will freeze")
        self.assertGreater(cache.report()["agings"], 0)

    def test_expert_keys_are_recognised_as_experts(self):
        self.assertTrue(is_expert(expert_kind(4)))
        self.assertFalse(is_expert("dense"))


class TestPlacementRanking(unittest.TestCase):
    def test_ranking_is_by_savings_per_resident_byte(self):
        small_hot = PinCandidate("small_hot", 1 * MB, accesses_per_token=1.0)
        big_hot = PinCandidate("big_hot", 100 * MB, accesses_per_token=1.0)
        self.assertEqual([c.key for c in rank([big_hot, small_hot])], ["small_hot", "big_hot"])

    def test_a_class_boundary_beats_any_value_density(self):
        """An enormously popular expert must not displace the attention block."""
        attention = PinCandidate("attn", 50 * MB, priority=CLASS_ALWAYS, accesses_per_token=1.0)
        expert = PinCandidate("expert", 1 * MB, priority=CLASS_ROUTED, accesses_per_token=100.0)
        self.assertEqual([c.key for c in rank([expert, attention])], ["attn", "expert"])

    def test_the_moe_priority_order_is_always_then_shared_then_routed(self):
        candidates = [
            PinCandidate("routed", 1 * MB, priority=CLASS_ROUTED, accesses_per_token=9.0),
            PinCandidate("shared", 1 * MB, priority=CLASS_SHARED),
            PinCandidate("attn", 1 * MB, priority=CLASS_ALWAYS),
        ]
        self.assertEqual([c.key for c in rank(candidates)], ["attn", "shared", "routed"])

    def test_a_moe_budget_buys_attention_before_experts(self):
        """Spending it on whole layers instead is what this ordering exists to prevent."""
        candidates = [PinCandidate(f"attn{i}", 10 * MB, priority=CLASS_ALWAYS) for i in range(4)]
        candidates += [PinCandidate(f"e{i}", 40 * MB, priority=CLASS_ROUTED,
                                    accesses_per_token=0.2) for i in range(20)]
        plan = plan_pins(candidates, 60 * MB)
        self.assertTrue(all(key.startswith("attn") for key in plan.pinned), plan.pinned)

    def test_hot_routed_experts_outrank_cold_ones(self):
        candidates = [PinCandidate(f"e{i}", 10 * MB, priority=CLASS_ROUTED,
                                   accesses_per_token=i / 10) for i in range(1, 6)]
        plan = plan_pins(candidates, 20 * MB)
        self.assertEqual(set(plan.pinned), {"e5", "e4"})

    def test_a_candidate_that_does_not_fit_does_not_stop_the_fill(self):
        candidates = [
            PinCandidate("huge", 100 * MB, accesses_per_token=1.0),
            PinCandidate("small", 5 * MB, accesses_per_token=1.0),
        ]
        plan = plan_pins(candidates, 50 * MB)
        self.assertEqual(plan.pinned, ("small",))
        self.assertEqual(len(plan.skipped), 1)

    def test_the_plan_never_exceeds_its_budget(self):
        candidates = [PinCandidate(f"k{i}", 7 * MB) for i in range(50)]
        for budget in (0, 1 * MB, 20 * MB, 100 * MB, 10_000 * MB):
            with self.subTest(budget=budget):
                plan = plan_pins(candidates, budget)
                self.assertLessEqual(plan.bytes_pinned, budget)

    def test_ranking_is_reproducible_for_equal_candidates(self):
        candidates = [PinCandidate(f"k{i}", 10 * MB) for i in range(10)]
        self.assertEqual([c.key for c in rank(candidates)],
                         [c.key for c in rank(list(reversed(candidates)))])

    def test_the_window_is_reserved_before_anything_is_pinned(self):
        self.assertEqual(pin_budget_from(100 * MB, 40 * MB), 60 * MB)
        self.assertEqual(pin_budget_from(30 * MB, 40 * MB), 0, "a committed window may zero the "
                                                               "pin budget, and that is allowed")


class TestReplanDamping(unittest.TestCase):
    def test_a_small_budget_shift_does_not_rebuild_the_plan(self):
        planner = PinPlanner(replan_bytes=10 * MB, window_budget_bytes=0)
        candidates = [PinCandidate(f"k{i}", 10 * MB) for i in range(10)]
        planner.build(candidates, 50 * MB)
        self.assertIsNone(planner.budget_changed(50 * MB, 52 * MB))
        self.assertEqual(planner.replans, 0)
        self.assertEqual(planner.suppressed, 1)

    def test_a_large_budget_shift_does_rebuild_it(self):
        planner = PinPlanner(replan_bytes=10 * MB, window_budget_bytes=0)
        candidates = [PinCandidate(f"k{i}", 10 * MB) for i in range(10)]
        planner.build(candidates, 50 * MB)
        plan = planner.budget_changed(50 * MB, 20 * MB)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.pinned), 2)
        self.assertEqual(planner.replans, 1)

    def test_a_shift_that_changes_nothing_reports_no_change(self):
        """Rebuilding is cheap; evicting because the answer is 'the same' is not."""
        planner = PinPlanner(replan_bytes=1 * MB, window_budget_bytes=0)
        candidates = [PinCandidate("k0", 10 * MB)]
        planner.build(candidates, 100 * MB)
        self.assertIsNone(planner.budget_changed(100 * MB, 90 * MB))
        self.assertEqual(planner.replans, 0)

    def test_the_planner_survives_the_budget_collapsing_to_zero(self):
        planner = PinPlanner(replan_bytes=1 * MB, window_budget_bytes=20 * MB)
        candidates = [PinCandidate(f"k{i}", 10 * MB) for i in range(4)]
        planner.build(candidates, 100 * MB)
        plan = planner.budget_changed(100 * MB, 0)
        self.assertIsNotNone(plan)
        self.assertTrue(plan.is_pure_streaming)


class TestReporting(unittest.TestCase):
    def test_the_report_carries_what_the_bench_harness_needs(self):
        cache, _ = build_cache(pinned=(dense_key(0),), host_bytes=2 * 10 * MB)
        walk(cache, [dense_key(i) for i in range(4)] * 2)
        report = cache.report()
        for key in ("hits_device", "hits_host", "misses", "hit_rate", "device_hit_rate",
                    "host_hit_rate", "evicted_to_host", "evicted_to_storage", "device_bytes",
                    "host_bytes", "window", "pinned", "fetches"):
            self.assertIn(key, report)

    def test_hits_and_misses_account_for_every_acquire(self):
        cache, _ = build_cache(host_bytes=4 * 10 * MB)
        keys = [dense_key(i) for i in range(4)] * 3
        walk(cache, keys)
        report = cache.report()
        self.assertEqual(report["hits_device"] + report["hits_host"] + report["misses"], len(keys))

    def test_resizing_the_device_evicts_down_to_the_new_capacity(self):
        cache, _ = build_cache(device_bytes=5 * 10 * MB, window=5)
        walk(cache, [dense_key(i) for i in range(5)])
        cache.resize_device(2 * 10 * MB)
        self.assertLessEqual(cache.report()["device_bytes"], 2 * 10 * MB)


if __name__ == "__main__":
    unittest.main(verbosity=2)
