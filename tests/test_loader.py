"""How many shard files stay memory-mapped, and why that is a hardware question.

A ``safe_open`` handle looks free. It is free in file descriptors and it is not free in address
space: the handle holds the shard mapped, and an operating system that charges mappings against a
commit limit is charged the shard's whole size the moment it is opened -- touched or not. Windows
charges RAM plus page file that way; Linux, under the default overcommit policy, charges a
read-only file mapping nothing at all, because it is page cache.

That difference had a concrete consequence. Sizing a checkpoint reads every shard's header once, so
the loader ended up holding every shard mapped at once, and a 67GB model died inside ``safe_open``
on a machine with an 18GB commit limit -- while doing nothing but reading headers, on hardware that
streams that model perfectly well.

What is tested here is the decision, not the platform. Every case below drives the loader with a
synthesised measurement, so the constrained path is exercised on a machine that is not constrained
and the unbounded path on one that is. Real measurements are only checked for shape and internal
consistency, since their values belong to whatever box the suite is running on.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocketllm.hw import caps  # noqa: E402
from rocketllm.hw.profile import DEFAULT_POLICY, Derivation, HardwareProfile  # noqa: E402
from rocketllm.streaming.loader import LayerLoader  # noqa: E402


def fake_profile(budget_bytes, io_workers=1, floor=2, ceiling=4):
    """A stand-in carrying only what the loader reads out of a profile."""

    class _Profile:
        derived = {
            "io_workers": Derivation(io_workers, "test", {}),
            "shard_mapping_budget_bytes": Derivation(int(budget_bytes), "test", {}),
            "shard_handle_limit": Derivation(
                ceiling, "test", {"shard_handle_floor": floor, "shard_handle_ceiling": ceiling}),
        }

    return _Profile()


def build_shards(root, count, values=1024):
    """A checkpoint of `count` one-module shards, named the way the splitter names them."""
    for index in range(count):
        save_file({f"model.layers.{index}.weight": torch.zeros(values, dtype=torch.float32)},
                  str(Path(root) / f"model.layers.{index}.safetensors"))
    return sorted(Path(root).glob("*.safetensors"))


class LoaderCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # Registered first so it runs LAST: cleanups unwind in reverse, and a mapped shard cannot
        # be deleted on Windows, so every loader has to be closed before the directory goes. That
        # ordering is not incidental tidiness -- it is the same property this file is testing.
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.shards = build_shards(self.root, 8)
        self.shard_bytes = self.shards[0].stat().st_size

    def loader(self, **kwargs):
        loader = LayerLoader(self.root, pool=None, **kwargs)
        self.addCleanup(loader.close)
        return loader


# ---- the measurement ---------------------------------------------------------------------------

class TestCommitHeadroom(unittest.TestCase):
    """The probe itself. Its numbers belong to this machine, so only their shape is asserted."""

    def test_it_answers_without_raising_on_whatever_this_machine_is(self):
        headroom = caps.commit_headroom()
        self.assertIsInstance(headroom.source, str)
        self.assertTrue(headroom.source)
        self.assertIsInstance(headroom.charges_mappings, bool)

    def test_a_measured_headroom_is_internally_consistent(self):
        headroom = caps.commit_headroom()
        if not headroom.measured:
            self.skipTest("this machine reports no commit accounting")
        self.assertGreaterEqual(headroom.available, 0)
        if headroom.total is not None:
            self.assertLessEqual(headroom.available, headroom.total)
        self.assertIn("GB", headroom.describe())

    def test_windows_reports_a_charged_commit_limit(self):
        """The platform this exists for. Elsewhere the case is skipped rather than asserted away."""
        if os.name != "nt":
            self.skipTest("not Windows")
        headroom = caps.commit_headroom()
        self.assertTrue(headroom.measured)
        self.assertTrue(headroom.charges_mappings)
        self.assertGreater(headroom.total, 0)

    def test_it_degrades_to_host_ram_rather_than_raising(self):
        """The degradation rule: an unreadable commit limit is not a reason to refuse to load."""
        original_nt, original_posix = caps._windows_commit, caps._linux_commit
        caps._windows_commit = caps._linux_commit = lambda: None
        try:
            headroom = caps.commit_headroom()
        finally:
            caps._windows_commit, caps._linux_commit = original_nt, original_posix
        self.assertIsInstance(headroom.source, str)
        if headroom.measured:
            self.assertGreaterEqual(headroom.available, 0)

    def test_the_profile_carries_the_measurement_and_derives_a_budget(self):
        profile = HardwareProfile.probe(device="cpu", storage_budget_seconds=0.05)
        self.assertIn("shard_mapping_budget_bytes", profile.derived)
        self.assertIn("shard_handle_limit", profile.derived)
        budget = profile.derived["shard_mapping_budget_bytes"].value
        self.assertGreaterEqual(budget, 0)
        if profile.commit_charges_mappings and profile.commit_available_bytes:
            # A share of what was measured, never more than it.
            self.assertLessEqual(budget, profile.commit_available_bytes)
            self.assertEqual(
                budget,
                int(profile.commit_available_bytes * DEFAULT_POLICY.shard_mapping_fraction))
        else:
            self.assertEqual(budget, 0)


# ---- choosing a mode ----------------------------------------------------------------------------

class TestModeSelection(LoaderCase):

    def test_a_checkpoint_that_fits_the_budget_keeps_every_handle_open(self):
        loader = self.loader(profile=fake_profile(budget_bytes=10 * 1024 ** 3))
        self.assertEqual(loader.handle_mode, "unbounded")
        self.assertEqual(loader.handle_limit, 0)
        self.assertIn("fits", loader.handle_reason)

    def test_a_checkpoint_larger_than_the_budget_is_bounded(self):
        loader = self.loader(profile=fake_profile(budget_bytes=self.shard_bytes * 2))
        self.assertEqual(loader.handle_mode, "bounded")
        self.assertGreaterEqual(loader.handle_limit, 2)
        self.assertLessEqual(loader.handle_limit, 4)

    def test_a_machine_that_does_not_charge_for_mappings_is_never_bounded(self):
        """Linux under default overcommit publishes no budget, and gets the fast path for free."""
        loader = self.loader(profile=fake_profile(budget_bytes=0))
        self.assertEqual(loader.handle_mode, "unbounded")
        self.assertIn("commit limit", loader.handle_reason)

    def test_no_profile_at_all_is_unbounded(self):
        self.assertEqual(self.loader().handle_mode, "unbounded")

    def test_the_bound_is_shared_across_reader_threads_not_applied_per_thread(self):
        """Four workers each holding four handles is sixteen mappings, not four.

        The budget below affords five of these shards against a checkpoint of eight, so the limit
        lands on the ceiling and there is room for the division to be visible.
        """
        loader = self.loader(profile=fake_profile(budget_bytes=self.shard_bytes * 5, io_workers=4))
        self.assertEqual(loader.handle_limit, 4)
        self.assertEqual(loader._per_thread_limit, 1)

    def test_the_count_is_what_the_budget_affords_when_that_is_under_the_ceiling(self):
        loader = self.loader(profile=fake_profile(budget_bytes=self.shard_bytes * 3))
        self.assertEqual(loader.handle_limit, 3)

    def test_a_limit_below_the_reader_count_is_raised_to_what_can_be_held(self):
        """A reader cannot hold fewer than the handle it is reading through, so two handles across
        four readers is not achievable. Reporting 2 while holding 4 would be the wrong answer."""
        loader = self.loader(shard_handle_limit=2,
                             profile=fake_profile(budget_bytes=self.shard_bytes * 5, io_workers=4))
        self.assertEqual(loader._per_thread_limit, 1)
        self.assertEqual(loader.handle_limit, 4)
        self.assertIn("cannot share fewer", loader.handle_reason)

    def test_a_budget_that_affords_less_than_the_floor_still_gets_the_floor(self):
        """One handle would reopen the same shard between planning it and reading it. Two is the
        smallest number that is not simply wasteful."""
        loader = self.loader(profile=fake_profile(budget_bytes=1))
        self.assertEqual(loader.handle_limit, 2)

    def test_an_explicit_count_overrides_the_measurement(self):
        loader = self.loader(profile=fake_profile(budget_bytes=10 * 1024 ** 3),
                             shard_handle_limit=2)
        self.assertEqual((loader.handle_mode, loader.handle_limit), ("bounded", 2))

    def test_unbounded_can_be_forced_on_a_machine_that_would_bound(self):
        loader = self.loader(profile=fake_profile(budget_bytes=self.shard_bytes * 2),
                             shard_handle_limit="unbounded")
        self.assertEqual((loader.handle_mode, loader.handle_limit), ("unbounded", 0))

    def test_a_meaningless_setting_is_refused_rather_than_guessed_at(self):
        with self.assertRaises(ValueError) as caught:
            LayerLoader(self.root, pool=None, shard_handle_limit="sometimes")
        self.assertIn("auto", str(caught.exception))

    def test_the_footprint_and_largest_shard_are_measured_from_the_checkpoint(self):
        loader = self.loader()
        self.assertEqual(loader.checkpoint_bytes(),
                         sum(p.stat().st_size for p in self.shards))
        self.assertEqual(loader.largest_shard_bytes(), self.shard_bytes)


# ---- eviction -----------------------------------------------------------------------------------

class TestHandleEviction(LoaderCase):

    def open_all(self, loader):
        for index in range(len(self.shards)):
            loader.plan(f"model.layers.{index}")
        return loader._local.handles

    def test_bounded_mode_never_holds_more_than_its_limit(self):
        loader = self.loader(profile=fake_profile(budget_bytes=self.shard_bytes * 2))
        handles = self.open_all(loader)
        self.assertLessEqual(len(handles), loader._per_thread_limit)
        self.assertEqual(loader.handle_opens, len(self.shards))
        self.assertEqual(loader.handle_evictions, len(self.shards) - loader._per_thread_limit)

    def test_unbounded_mode_holds_them_all(self):
        loader = self.loader(profile=fake_profile(budget_bytes=10 * 1024 ** 3))
        handles = self.open_all(loader)
        self.assertEqual(len(handles), len(self.shards))
        self.assertEqual(loader.handle_evictions, 0)

    def test_the_least_recently_used_shard_is_the_one_dropped(self):
        loader = self.loader(shard_handle_limit=2)
        loader.plan("model.layers.0")
        loader.plan("model.layers.1")
        loader.plan("model.layers.0")      # 0 becomes the most recent, so 1 is next out
        loader.plan("model.layers.2")
        held = {Path(path).stem for path in loader._local.handles}
        self.assertEqual(held, {"model.layers.0", "model.layers.2"})

    def test_a_repeated_shard_is_not_reopened(self):
        loader = self.loader(shard_handle_limit=2)
        for _ in range(5):
            loader.plan("model.layers.0")
        self.assertEqual(loader.handle_opens, 1)
        self.assertEqual(loader.handle_evictions, 0)

    def test_an_evicted_handle_is_closed_rather_than_merely_dropped(self):
        """Dropping the reference leaves the mapping alive until the collector runs, which is the
        whole thing this exists to stop."""
        loader = self.loader(shard_handle_limit=1)
        loader.plan("model.layers.0")
        evicted = next(iter(loader._local.handles.values()))
        loader.plan("model.layers.1")
        self.assertEqual(loader.handle_evictions, 1)
        with self.assertRaises(Exception):
            evicted.get_slice("model.layers.0.weight")

    def test_close_releases_everything_still_held(self):
        loader = LayerLoader(self.root, pool=None, shard_handle_limit="unbounded")
        for index in range(len(self.shards)):
            loader.plan(f"model.layers.{index}")
        self.assertEqual(len(loader._handles), len(self.shards))
        loader.close()
        self.assertEqual(loader._handles, set())

    def test_the_shards_can_be_deleted_after_close(self):
        """A mapped file cannot be unlinked on Windows, so this is the observable proof that the
        mappings are gone rather than merely dereferenced."""
        loader = LayerLoader(self.root, pool=None, shard_handle_limit="unbounded")
        for index in range(len(self.shards)):
            loader.plan(f"model.layers.{index}")
        loader.close()
        for shard in self.shards:
            shard.unlink()
        self.assertEqual(list(self.root.glob("*.safetensors")), [])


# ---- what must not change -----------------------------------------------------------------------

class TestReadsAreUnaffected(LoaderCase):
    """Bounding handles must change what stays open and nothing else."""

    def test_a_bounded_loader_plans_exactly_what_an_unbounded_one_does(self):
        bounded = self.loader(shard_handle_limit=2)
        unbounded = self.loader(shard_handle_limit="unbounded")
        for index in range(len(self.shards)):
            layout = bounded.plan(f"model.layers.{index}")
            other = unbounded.plan(f"model.layers.{index}")
            self.assertEqual(layout.total_bytes, other.total_bytes)
            self.assertEqual([p.name for p in layout.placements],
                             [p.name for p in other.placements])
            self.assertEqual([p.shape for p in layout.placements],
                             [p.shape for p in other.placements])

    def test_a_shard_reopened_after_eviction_still_reads_the_same_bytes(self):
        loader = self.loader(shard_handle_limit=1)
        first = loader.plan("model.layers.0")
        loader.plan("model.layers.1")          # evicts 0
        again = loader.plan("model.layers.0")  # reopens it
        self.assertEqual(first.total_bytes, again.total_bytes)
        self.assertGreaterEqual(loader.handle_evictions, 1)

    def test_stats_report_the_mode_so_a_slow_run_can_be_explained(self):
        stats = self.loader(shard_handle_limit=2).stats()
        for key in ("shard_handle_mode", "shard_handle_limit", "shard_handle_opens",
                    "shard_handle_evictions", "io_workers", "reads", "bytes_read"):
            self.assertIn(key, stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
