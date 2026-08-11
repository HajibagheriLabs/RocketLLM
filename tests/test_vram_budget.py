"""Tests for the live device budget, with every memory call mocked.

The budget is the number every placement decision is made against, and it is wrong in a way that
does not raise: an over-claimed budget shows up as an out-of-memory error several layers later, and
an under-claimed one just makes the run slow. Neither points back here. So the arithmetic is pinned
by hand-checked numbers rather than by whatever the machine running the suite happens to report.

Nothing here needs an accelerator. The whole point of the exercise is that the interesting cases --
a backend that cannot report allocator counters, a card whose free memory is collapsing under a
growing KV cache -- are not reproducible on any one developer's hardware anyway.
"""
import unittest

import torch

from rocketllm.hw.caps import MemoryReport
from rocketllm.memory.budget import AllocatorSetup, BudgetSample, VramBudget

GB = 1024 ** 3
MB = 1024 ** 2


class FakeCaps:
    """Stands in for a DeviceCaps, serving readings from a script.

    Only `memory()` matters: it is the entire surface the budget uses to measure, which is itself
    the point -- everything backend-specific already lives behind the device abstraction.
    """

    backend = "cuda"

    def __init__(self, readings, device="cuda:0"):
        self.device = torch.device(device)
        self._readings = list(readings)
        self.calls = 0

    def memory(self, reserve_bytes=0):
        self.calls += 1
        # The last reading repeats, so a test only has to script what it cares about.
        free, reserved, allocated, estimated = self._readings[
            min(self.calls - 1, len(self._readings) - 1)]
        if reserved is None or allocated is None:
            budget = max(0, (free or 0) - reserve_bytes)
            return MemoryReport(total=24 * GB, free=free, reserved=None, allocated=None,
                                budget=budget, estimated=True, note="conservative")
        held = max(0, reserved - allocated)
        return MemoryReport(total=24 * GB, free=free, reserved=reserved, allocated=allocated,
                            budget=max(0, free + held - reserve_bytes), estimated=estimated,
                            note="free + (reserved - allocated) - reserve")


class FakeDerivation:
    def __init__(self, value):
        self.value = value


class FakeProfile:
    """A profile stub.

    `hysteresis_bytes` is the absolute escape hatch and pins the band wherever it is given, which is
    what most of these tests want because it makes the arithmetic hand-checkable. `hysteresis_ratio`
    is what a real profile derives, and makes the band a share of the budget in play.
    """

    def __init__(self, reserve=0, hysteresis_bytes=None, hysteresis_samples=1,
                 hysteresis_ratio=None):
        self.derived = {
            "reserve_bytes": FakeDerivation(reserve),
            "budget_hysteresis_samples": FakeDerivation(hysteresis_samples),
        }
        if hysteresis_bytes is not None:
            self.derived["budget_hysteresis_bytes"] = FakeDerivation(hysteresis_bytes)
        if hysteresis_ratio is not None:
            self.derived["budget_hysteresis_ratio"] = FakeDerivation(hysteresis_ratio)


def build(readings, profile=None, **kwargs):
    """A budget wired to scripted readings, with allocator configuration kept out of the way."""
    kwargs.setdefault("configure_allocator_env", False)
    return VramBudget(device_caps=FakeCaps(readings), profile=profile, **kwargs)


def steady(free, reserved=0, allocated=0):
    return (free, reserved, allocated, False)


class TestTheArithmetic(unittest.TestCase):
    """free + (reserved - allocated) - reserve, and nothing else."""

    def test_held_blocks_are_added_back_to_what_the_driver_calls_free(self):
        """The allocator's freed-but-held blocks are allocatable; the driver disagrees, and is
        wrong from this process's point of view."""
        budget = build([steady(free=4 * GB, reserved=3 * GB, allocated=1 * GB)],
                       profile=FakeProfile(reserve=1 * GB))
        # 4 free + (3 reserved - 1 allocated) - 1 reserve = 5
        self.assertEqual(budget.current(), 5 * GB)

    def test_a_fully_used_allocator_pool_adds_nothing_back(self):
        budget = build([steady(free=4 * GB, reserved=2 * GB, allocated=2 * GB)],
                       profile=FakeProfile(reserve=1 * GB))
        self.assertEqual(budget.current(), 3 * GB)

    def test_the_sample_keeps_every_term_so_a_surprise_can_be_taken_apart(self):
        budget = build([steady(free=4 * GB, reserved=3 * GB, allocated=1 * GB)],
                       profile=FakeProfile(reserve=1 * GB))
        reading = budget.history[-1]
        self.assertIsInstance(reading, BudgetSample)
        self.assertEqual(reading.free, 4 * GB)
        self.assertEqual(reading.reserved, 3 * GB)
        self.assertEqual(reading.allocated, 1 * GB)
        self.assertEqual(reading.held, 2 * GB)
        self.assertEqual(reading.reserve, 1 * GB)
        self.assertEqual(reading.usable, 5 * GB)

    def test_a_reserve_larger_than_the_card_floors_at_zero_rather_than_going_negative(self):
        budget = build([steady(free=1 * GB, reserved=0, allocated=0)],
                       profile=FakeProfile(reserve=99 * GB))
        self.assertEqual(budget.current(), 0)
        self.assertEqual(budget.target(), 0)


class TestReserveComesFromTheProfile(unittest.TestCase):
    """Not a constant, and not a fraction picked by hand."""

    def test_the_profiles_reserve_is_what_gets_subtracted(self):
        for reserve in (0, 512 * 1024 * 1024, 2 * GB):
            with self.subTest(reserve=reserve):
                budget = build([steady(free=8 * GB)], profile=FakeProfile(reserve=reserve))
                self.assertEqual(budget.reserve_bytes, reserve)
                self.assertEqual(budget.current(), 8 * GB - reserve)

    def test_two_machines_with_the_same_free_memory_get_different_budgets(self):
        """The whole reason reserve is measured: identical readings, different allocator behaviour."""
        small = build([steady(free=8 * GB)], profile=FakeProfile(reserve=256 * 1024 * 1024))
        large = build([steady(free=8 * GB)], profile=FakeProfile(reserve=3 * GB))
        self.assertGreater(small.target(), large.target())

    def test_an_explicit_reserve_overrides_the_profile(self):
        budget = build([steady(free=8 * GB)], profile=FakeProfile(reserve=3 * GB),
                       reserve_bytes=1 * GB)
        self.assertEqual(budget.reserve_bytes, 1 * GB)

    def test_no_profile_degrades_loudly_rather_than_inventing_a_number(self):
        with self.assertLogs("rocketllm.hw.caps", level="INFO") as captured:
            budget = build([steady(free=8 * GB)], profile=None)
        self.assertEqual(budget.reserve_bytes, 0)
        self.assertIn("no hardware profile", "\n".join(captured.output))


class TestConservativeFallback(unittest.TestCase):
    """A backend that cannot report the components must under-claim, and say so."""

    def test_a_backend_without_allocator_counters_adds_nothing_back(self):
        budget = build([(4 * GB, None, None, True)], profile=FakeProfile(reserve=1 * GB))
        self.assertEqual(budget.current(), 3 * GB)
        self.assertEqual(budget.history[-1].held, 0)
        self.assertTrue(budget.history[-1].estimated)

    def test_the_fallback_is_announced(self):
        import rocketllm.hw.caps as C

        C.reset_announcements()
        with self.assertLogs("rocketllm.hw.caps", level="INFO") as captured:
            build([(4 * GB, None, None, True)], profile=FakeProfile())
        message = "\n".join(captured.output)
        self.assertIn("conservative", message)
        self.assertIn("under-claim", message)

    def test_the_estimated_reading_is_never_larger_than_the_exact_one_would_be(self):
        """Under-claiming is the safe direction; over-claiming surfaces as an OOM later."""
        exact = build([steady(free=4 * GB, reserved=3 * GB, allocated=1 * GB)],
                      profile=FakeProfile(reserve=1 * GB))
        estimated = build([(4 * GB, None, None, True)], profile=FakeProfile(reserve=1 * GB))
        self.assertLess(estimated.current(), exact.current())


class TestHysteresis(unittest.TestCase):
    """A cache that reorganises on noise pays a full streaming pass to learn nothing."""

    def profile(self):
        return FakeProfile(reserve=0, hysteresis_bytes=1 * GB, hysteresis_samples=3)

    def test_a_single_sample_spike_does_not_move_the_target(self):
        budget = build([steady(8 * GB), steady(2 * GB), steady(8 * GB), steady(8 * GB)],
                       profile=self.profile())
        self.assertEqual(budget.target(), 8 * GB)
        for _ in range(3):
            budget.sample()
        self.assertEqual(budget.target(), 8 * GB, "one outlier moved the published budget")
        self.assertEqual(budget.changes, 0)

    def test_a_sustained_shift_does_move_the_target(self):
        budget = build([steady(8 * GB)] + [steady(2 * GB)] * 5, profile=self.profile())
        self.assertEqual(budget.target(), 8 * GB)
        for _ in range(3):
            budget.sample()
        self.assertEqual(budget.target(), 2 * GB)
        self.assertEqual(budget.changes, 1)

    def test_the_shift_must_persist_for_the_full_count_not_one_sample_less(self):
        budget = build([steady(8 * GB)] + [steady(2 * GB)] * 5, profile=self.profile())
        budget.sample()
        budget.sample()
        self.assertEqual(budget.target(), 8 * GB, "moved before the streak was complete")
        budget.sample()
        self.assertEqual(budget.target(), 2 * GB)

    def test_a_move_inside_the_threshold_is_never_published_however_long_it_lasts(self):
        """Below the noise floor there is nothing to react to, however many samples agree."""
        budget = build([steady(8 * GB)] + [steady(8 * GB - 100 * 1024 * 1024)] * 10,
                       profile=self.profile())
        for _ in range(10):
            budget.sample()
        self.assertEqual(budget.target(), 8 * GB)
        self.assertEqual(budget.changes, 0)

    def test_jitter_across_the_threshold_in_both_directions_never_accumulates(self):
        """Alternating overshoots are the noise itself, not a trend to follow."""
        readings = [steady(8 * GB)]
        for _ in range(6):
            readings += [steady(2 * GB), steady(14 * GB)]
        budget = build(readings, profile=self.profile())
        for _ in range(12):
            budget.sample()
        self.assertEqual(budget.changes, 0, "alternating spikes were mistaken for a shift")

    def test_a_growing_budget_also_needs_the_streak(self):
        budget = build([steady(2 * GB)] + [steady(9 * GB)] * 5, profile=self.profile())
        self.assertEqual(budget.target(), 2 * GB)
        budget.sample()
        self.assertEqual(budget.target(), 2 * GB)
        budget.sample()
        budget.sample()
        self.assertEqual(budget.target(), 9 * GB)

    def test_the_published_value_is_the_most_conservative_of_the_streak(self):
        """Publishing the last reading of a shrinking run would claim room already gone."""
        budget = build([steady(8 * GB), steady(3 * GB), steady(2 * GB), steady(2500 * 1024 ** 2)],
                       profile=self.profile())
        for _ in range(3):
            budget.sample()
        self.assertEqual(budget.target(), 2 * GB)

    def test_the_first_reading_is_published_without_waiting_for_a_streak(self):
        """An unsampled budget is unknown, not zero, and zero would mean "cache nothing"."""
        budget = build([steady(6 * GB)], profile=self.profile())
        self.assertEqual(budget.target(), 6 * GB)

    def test_the_first_recorded_sample_carries_the_target_it_established(self):
        """A trace whose opening row says target=0 reads as a budget that started at nothing."""
        budget = build([steady(6 * GB)], profile=self.profile())
        self.assertEqual(budget.history[0].target, 6 * GB)

    def test_current_tracks_the_machine_while_target_holds_still(self):
        budget = build([steady(8 * GB), steady(2 * GB)], profile=self.profile())
        budget.sample()
        self.assertEqual(budget.current(), 2 * GB)
        self.assertEqual(budget.target(), 8 * GB)


class TestCallbackAndHistory(unittest.TestCase):
    def test_on_change_fires_once_per_published_move_with_both_values(self):
        seen = []
        budget = build([steady(8 * GB)] + [steady(2 * GB)] * 5,
                       profile=FakeProfile(hysteresis_bytes=1 * GB, hysteresis_samples=3),
                       on_change=lambda old, new, sample: seen.append((old, new)))
        for _ in range(5):
            budget.sample()
        self.assertEqual(seen, [(8 * GB, 2 * GB)])

    def test_no_callback_fires_when_nothing_is_published(self):
        seen = []
        budget = build([steady(8 * GB), steady(2 * GB), steady(8 * GB), steady(8 * GB)],
                       profile=FakeProfile(hysteresis_bytes=1 * GB, hysteresis_samples=3),
                       on_change=lambda *a: seen.append(a))
        for _ in range(3):
            budget.sample()
        self.assertEqual(seen, [])

    def test_the_history_buffer_records_every_sample_and_is_bounded(self):
        budget = build([steady(8 * GB)], profile=FakeProfile(), history=4)
        for _ in range(20):
            budget.sample()
        self.assertEqual(len(budget.history), 4)

    def test_the_trace_is_plain_data_for_the_bench_harness(self):
        budget = build([steady(8 * GB)], profile=FakeProfile())
        budget.sample()
        trace = budget.trace()
        self.assertTrue(all(isinstance(row, dict) for row in trace))
        for key in ("at", "free", "reserved", "allocated", "held", "usable", "target"):
            self.assertIn(key, trace[-1])

    def test_a_callback_that_raises_is_the_callers_problem_not_a_corrupt_budget(self):
        def explode(old, new, sample):
            raise RuntimeError("callback blew up")

        budget = build([steady(8 * GB)] + [steady(2 * GB)] * 5,
                       profile=FakeProfile(hysteresis_bytes=1 * GB, hysteresis_samples=3),
                       on_change=explode)
        for _ in range(2):
            budget.sample()
        with self.assertRaises(RuntimeError):
            budget.sample()
        # The published move still happened, and the lock was released on the way out.
        self.assertEqual(budget.target(), 2 * GB)
        self.assertEqual(budget.sample().usable, 2 * GB)


class TestGenerationBoundary(unittest.TestCase):
    def test_reset_republishes_at_once_because_the_kv_cache_just_went_away(self):
        budget = build([steady(2 * GB)] + [steady(12 * GB)] * 3,
                       profile=FakeProfile(hysteresis_bytes=1 * GB, hysteresis_samples=3))
        self.assertEqual(budget.target(), 2 * GB)
        budget.reset()
        self.assertEqual(budget.target(), 12 * GB)

    def test_reset_reports_the_move_through_the_callback(self):
        seen = []
        budget = build([steady(2 * GB)] + [steady(12 * GB)] * 3,
                       profile=FakeProfile(hysteresis_bytes=1 * GB, hysteresis_samples=3),
                       on_change=lambda old, new, sample: seen.append((old, new)))
        budget.reset()
        self.assertEqual(seen, [(2 * GB, 12 * GB)])


class TestTheBandScalesWithTheBudget(unittest.TestCase):
    """REGRESSION TEST. A band sized off the whole card cannot move a constrained budget.

    The failure this pins, measured on a 24GB card: the band came out at 1693MB, because it was
    derived as a fraction of the *whole card* (and floored at the measured fragmentation ratio times
    total memory). The budget the engine was actually working against, once the weights were
    resident, was ~507MB. So the band was three times the budget it governed, and no change of any
    kind could ever be published -- quantizing the KV cache genuinely freed 66MB and the pin plan
    never heard about it.

    The rule now is that the band is a share of the budget in play, so it means the same thing on a
    4GB card and a 192GB one. Hysteresis is not weakened: what is asserted below is that ordinary
    allocator noise is still rejected, and only a sustained, proportionally material move is
    published.
    """

    #: What the Prompt 13 fixture actually recovered by quantizing its context.
    KV_RECOVERED = 66 * MB
    CONSTRAINED = 507 * MB

    def test_the_old_whole_card_band_suppresses_a_real_kv_recovery(self):
        """The bug, reproduced: a band sized off the card swallows the whole effect."""
        readings = [steady(self.CONSTRAINED)] + [steady(self.CONSTRAINED + self.KV_RECOVERED)] * 6
        budget = build(readings, profile=FakeProfile(hysteresis_bytes=1693 * MB,
                                                     hysteresis_samples=3))
        for _ in range(6):
            budget.sample()
        self.assertGreater(budget.current(), self.CONSTRAINED, "the memory really was recovered")
        self.assertEqual(budget.target(), self.CONSTRAINED,
                         "the published target moved, so this no longer reproduces the failure")
        self.assertEqual(budget.changes, 0)

    def test_a_proportional_band_publishes_the_same_recovery(self):
        readings = [steady(self.CONSTRAINED)] + [steady(self.CONSTRAINED + self.KV_RECOVERED)] * 6
        seen = []
        budget = build(readings, profile=FakeProfile(hysteresis_ratio=0.065, hysteresis_samples=3),
                       on_change=lambda old, new, sample: seen.append((old, new)))
        for _ in range(6):
            budget.sample()
        self.assertEqual(budget.target(), self.CONSTRAINED + self.KV_RECOVERED)
        self.assertEqual(len(seen), 1, "the move should be published exactly once")

    def test_allocator_noise_is_still_rejected(self):
        """The band exists to stop the cache reshuffling over jitter. It still must."""
        noise = int(self.CONSTRAINED * 0.02)
        readings = [steady(self.CONSTRAINED)]
        for step in range(12):
            readings.append(steady(self.CONSTRAINED + (noise if step % 2 else -noise)))
        budget = build(readings, profile=FakeProfile(hysteresis_ratio=0.065, hysteresis_samples=3))
        for _ in range(12):
            budget.sample()
        self.assertEqual(budget.target(), self.CONSTRAINED)
        self.assertEqual(budget.changes, 0)

    def test_the_band_means_the_same_thing_on_any_card(self):
        """Scale-free by construction: the same relative move publishes at 1GB and at 100GB."""
        for base in (1 * GB, 10 * GB, 100 * GB):
            move = int(base * 0.10)
            budget = build([steady(base)] + [steady(base + move)] * 6,
                           profile=FakeProfile(hysteresis_ratio=0.065, hysteresis_samples=3))
            for _ in range(6):
                budget.sample()
            with self.subTest(base_gb=base // GB):
                self.assertEqual(budget.target(), base + move)

    def test_a_move_smaller_than_the_share_is_not_published_however_long_it_lasts(self):
        small = int(self.CONSTRAINED * 0.03)
        budget = build([steady(self.CONSTRAINED)] + [steady(self.CONSTRAINED + small)] * 20,
                       profile=FakeProfile(hysteresis_ratio=0.065, hysteresis_samples=3))
        for _ in range(20):
            budget.sample()
        self.assertEqual(budget.target(), self.CONSTRAINED)

    def test_the_band_shrinks_with_the_budget_it_governs(self):
        """The point of the change: a smaller budget gets a proportionally smaller band."""
        big = build([steady(20 * GB)], profile=FakeProfile(hysteresis_ratio=0.05))
        small = build([steady(500 * MB)], profile=FakeProfile(hysteresis_ratio=0.05))
        self.assertEqual(big.hysteresis_bytes, int(20 * GB * 0.05))
        self.assertEqual(small.hysteresis_bytes, int(500 * MB * 0.05))
        self.assertLess(small.hysteresis_bytes, big.hysteresis_bytes)

    def test_an_absolute_band_still_overrides_the_share(self):
        """The debugging escape hatch stays: an explicit byte count pins the band."""
        budget = build([steady(8 * GB)], profile=FakeProfile(hysteresis_ratio=0.5),
                       hysteresis_bytes=123 * MB)
        self.assertEqual(budget.hysteresis_bytes, 123 * MB)


class TestBothDirections(unittest.TestCase):
    """Growth and recovery must both obey hysteresis, and both must eventually publish."""

    BASE = 800 * MB

    def profile(self):
        return FakeProfile(hysteresis_ratio=0.06, hysteresis_samples=3)

    def test_a_growing_context_eventually_lowers_the_target(self):
        grown = self.BASE - 200 * MB
        seen = []
        budget = build([steady(self.BASE)] + [steady(grown)] * 6, profile=self.profile(),
                       on_change=lambda old, new, sample: seen.append((old, new)))
        for _ in range(6):
            budget.sample()
        self.assertEqual(budget.target(), grown)
        self.assertEqual(seen, [(self.BASE, grown)])

    def test_a_shrinking_context_eventually_raises_the_target(self):
        freed = self.BASE + 200 * MB
        seen = []
        budget = build([steady(self.BASE)] + [steady(freed)] * 6, profile=self.profile(),
                       on_change=lambda old, new, sample: seen.append((old, new)))
        for _ in range(6):
            budget.sample()
        self.assertEqual(budget.target(), freed)
        self.assertEqual(seen, [(self.BASE, freed)])

    def test_neither_direction_publishes_before_the_move_has_persisted(self):
        for delta in (+200 * MB, -200 * MB):
            budget = build([steady(self.BASE)] + [steady(self.BASE + delta)] * 6,
                           profile=self.profile())
            budget.sample()
            budget.sample()
            with self.subTest(delta_mb=delta // MB):
                self.assertEqual(budget.target(), self.BASE,
                                 "published before the streak was satisfied")

    def test_token_by_token_growth_does_not_republish_every_sample(self):
        """The reason hysteresis exists: a KV cache growing steadily must not thrash the cache."""
        readings = [steady(self.BASE - step * MB) for step in range(200)]
        budget = build(readings, profile=self.profile())
        for _ in range(199):
            budget.sample()
        self.assertLess(budget.changes, 8,
                        f"{budget.changes} republishes over 200 samples of steady growth is "
                        f"thrashing, which is what the band exists to prevent")
        self.assertGreater(budget.changes, 0, "the target has to follow a sustained trend at all")


class TestNoKvEstimator(unittest.TestCase):
    """Growth shows up because it is measured, which is why no size formula exists here."""

    def test_a_shrinking_card_is_tracked_with_no_knowledge_of_what_is_consuming_it(self):
        # Free memory falling as a KV cache grows, in even steps. Nothing tells the budget why.
        readings = [steady(free) for free in
                    (12 * GB, 10 * GB, 8 * GB, 6 * GB, 4 * GB, 2 * GB)]
        budget = build(readings, profile=FakeProfile(hysteresis_bytes=1 * GB,
                                                     hysteresis_samples=2))
        for _ in range(5):
            budget.sample()
        self.assertLess(budget.target(), 12 * GB)
        self.assertEqual(budget.current(), 2 * GB)

    def test_the_module_carries_no_kv_cache_size_model(self):
        """A modelled size would need updating for every new attention variant; a measurement
        does not."""
        import rocketllm.memory.budget as module

        source = open(module.__file__, encoding="utf-8").read().lower()
        for banned in ("num_key_value_heads", "head_dim", "n_kv_heads", "kv_bytes_per_token"):
            self.assertNotIn(banned, source,
                             f"{banned} suggests a modelled KV size crept in; it must be measured")


class TestAllocatorConfiguration(unittest.TestCase):
    def test_a_backend_without_the_setting_is_not_applicable_rather_than_an_error(self):
        from rocketllm.memory.budget import configure_allocator

        setup = configure_allocator(torch.device("cpu"))
        self.assertEqual(setup.status, "not_applicable")
        self.assertFalse(setup.effective)

    def test_declining_configuration_is_recorded_not_silently_skipped(self):
        budget = build([steady(8 * GB)], profile=FakeProfile())
        self.assertIsInstance(budget.allocator, AllocatorSetup)
        self.assertEqual(budget.allocator.status, "not_applicable")

    def test_an_existing_setting_is_respected(self):
        import os

        from rocketllm.memory.budget import configure_allocator

        saved = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        try:
            setup = configure_allocator(torch.device("cuda:0"))
        finally:
            if saved is None:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = saved
        # On a machine with no CUDA the backend check wins first; either answer is correct, and
        # neither may claim the setting was newly applied.
        self.assertIn(setup.status, ("already_configured", "not_applicable"))
        self.assertFalse(setup.effective)


class TestSummary(unittest.TestCase):
    def test_the_summary_explains_the_budget_it_is_publishing(self):
        budget = build([steady(free=4 * GB, reserved=3 * GB, allocated=1 * GB)],
                       profile=FakeProfile(reserve=1 * GB, hysteresis_bytes=1 * GB,
                                           hysteresis_samples=3))
        summary = budget.summary()
        for key in ("backend", "reserve_bytes", "hysteresis_bytes", "hysteresis_samples",
                    "target_bytes", "samples", "changes", "allocator"):
            self.assertIn(key, summary)
        self.assertEqual(summary["reserve_bytes"], 1 * GB)
        self.assertEqual(summary["target_bytes"], 5 * GB)


if __name__ == "__main__":
    unittest.main(verbosity=2)
