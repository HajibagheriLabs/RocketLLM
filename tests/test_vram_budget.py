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
    def __init__(self, reserve=0, hysteresis_bytes=0, hysteresis_samples=1):
        self.derived = {
            "reserve_bytes": FakeDerivation(reserve),
            "budget_hysteresis_bytes": FakeDerivation(hysteresis_bytes),
            "budget_hysteresis_samples": FakeDerivation(hysteresis_samples),
        }


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
