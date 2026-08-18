"""How much of a checkpoint may be held memory-mapped, and why that is a hardware question.

A ``safe_open`` handle looks free. It is free in file descriptors and it is not free in address
space: the handle holds the shard mapped, and an operating system that charges mappings against a
commit limit is charged the shard's whole size the moment it is opened -- touched or not. Windows
charges RAM plus page file that way; Linux, under the default overcommit policy, charges a
read-only file mapping nothing at all, because it is page cache.

That difference had a concrete consequence. safetensors returns tensors that ALIAS the mapping, so
every layer the cache held kept its whole shard mapped, and a 67GB model exhausted an 18GB commit
limit on hardware that streams it perfectly well.

So the engine picks how to read a shard from what it measured: mapping is the cheaper route and is
kept wherever the checkpoint fits what the machine will let the process charge, and byte-range
reads are used where it does not. What is tested here is that decision, not the platform. Every
case below drives it with a synthesised measurement, so the constrained path is exercised on a
machine that is not constrained and the unconstrained path on one that is. Real measurements are
checked only for shape and internal consistency, since their values belong to whatever box the
suite is running on. That the two routes return the same bytes is ``tests/test_shards.py``.
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
from rocketllm.streaming import shards  # noqa: E402
from rocketllm.streaming.loader import LayerLoader  # noqa: E402


def fake_profile(budget_bytes, io_workers=1, floor=2, ceiling=4):
    """A stand-in carrying only what the reader and the loader read out of a profile."""

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
        # be deleted on Windows, so every reader has to be released before the directory goes.
        # That ordering is not incidental tidiness -- it is the same property this file is testing.
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(shards.release_all)
        self.root = Path(self._tmp.name)
        self.shards = build_shards(self.root, 8)
        self.shard_bytes = self.shards[0].stat().st_size

    def reader(self, mapped=False, **kwargs):
        """A reader for this checkpoint, optionally forced onto the mapped path."""
        original = shards.direct_reads_available
        if mapped:
            shards.direct_reads_available = lambda: False
        try:
            built = shards.ShardReader(self.root, **kwargs)
        finally:
            shards.direct_reads_available = original
        self.addCleanup(built.release)
        return built

    def loader(self, **kwargs):
        loader = LayerLoader(self.root, pool=None, **kwargs)
        self.addCleanup(loader.close)
        return loader

    def read_all(self, reader):
        """Read one tensor out of every shard, which is what opens a handle on the mapped path."""
        for index in range(len(self.shards)):
            reader.read_tensors(self.root / f"model.layers.{index}.safetensors")
        return getattr(reader._local, "handles", {})


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

    def test_a_checkpoint_that_fits_the_budget_is_mapped(self):
        """Mapping is the cheaper read -- no copy at all -- so it is kept wherever it is safe."""
        reader = self.reader(profile=fake_profile(budget_bytes=10 * 1024 ** 3))
        self.assertEqual(reader.mode, shards.ShardReader.MAPPED)
        self.assertEqual(reader.handle_limit, 0)
        self.assertIn("fits", reader.reason)

    def test_a_checkpoint_larger_than_the_budget_is_read_by_byte_range(self):
        """Bounding handles cannot fix this. safetensors hands back tensors that alias the
        mapping, so a cached expert holds its whole shard however few handles are open."""
        reader = self.reader(profile=fake_profile(budget_bytes=self.shard_bytes * 2))
        self.assertEqual(reader.mode, shards.ShardReader.DIRECT)
        self.assertEqual(reader.handle_limit, 0)
        self.assertIn("byte range", reader.reason)

    def test_a_machine_that_does_not_charge_for_mappings_maps_freely(self):
        """Linux under default overcommit publishes no budget, and keeps the fast path for free."""
        reader = self.reader(profile=fake_profile(budget_bytes=0))
        self.assertEqual(reader.mode, shards.ShardReader.MAPPED)
        self.assertIn("commit limit", reader.reason)

    def test_no_profile_at_all_maps(self):
        self.assertEqual(self.reader().mode, shards.ShardReader.MAPPED)

    def test_a_host_that_cannot_read_the_payload_as_stored_maps_and_bounds_instead(self):
        """Big-endian. It has no direct path, so the handle bound is all that is left."""
        reader = self.reader(mapped=True, profile=fake_profile(budget_bytes=self.shard_bytes * 3))
        self.assertEqual(reader.mode, shards.ShardReader.MAPPED)
        self.assertEqual(reader.handle_limit, 3)
        self.assertIn("big-endian", reader.reason)

    def test_the_bound_is_shared_across_reader_threads_not_applied_per_thread(self):
        """Four readers each holding four handles is sixteen mappings, not four.

        The budget below affords five of these shards against a checkpoint of eight, so the limit
        lands on the ceiling and there is room for the division to be visible.
        """
        reader = self.reader(mapped=True, readers=4,
                             profile=fake_profile(budget_bytes=self.shard_bytes * 5))
        self.assertEqual(reader.handle_limit, 4)
        self.assertEqual(reader._per_thread_limit, 1)

    def test_a_limit_below_the_reader_count_is_raised_to_what_can_be_held(self):
        """A reader cannot hold fewer than the handle it is reading through, so two handles across
        four readers is not achievable. Reporting 2 while holding 4 would be the wrong answer."""
        reader = self.reader(shard_handle_limit=2, readers=4,
                             profile=fake_profile(budget_bytes=self.shard_bytes * 5))
        self.assertEqual(reader._per_thread_limit, 1)
        self.assertEqual(reader.handle_limit, 4)
        self.assertIn("cannot share fewer", reader.reason)

    def test_a_budget_that_affords_less_than_the_floor_still_gets_the_floor(self):
        """One handle would reopen the same shard between two reads of it. Two is the smallest
        number that is not simply wasteful."""
        reader = self.reader(mapped=True, profile=fake_profile(budget_bytes=1))
        self.assertEqual(reader.handle_limit, 2)

    def test_an_explicit_count_overrides_the_measurement(self):
        reader = self.reader(profile=fake_profile(budget_bytes=self.shard_bytes * 2),
                             shard_handle_limit=2)
        self.assertEqual((reader.mode, reader.handle_limit),
                         (shards.ShardReader.MAPPED, 2))

    def test_mapping_can_be_forced_on_a_machine_that_would_not(self):
        reader = self.reader(profile=fake_profile(budget_bytes=self.shard_bytes * 2),
                             shard_handle_limit="unbounded")
        self.assertEqual((reader.mode, reader.handle_limit),
                         (shards.ShardReader.MAPPED, 0))

    def test_byte_range_reads_can_be_forced_on_a_machine_that_would_map(self):
        reader = self.reader(profile=fake_profile(budget_bytes=10 * 1024 ** 3),
                             shard_handle_limit="direct")
        self.assertEqual(reader.mode, shards.ShardReader.DIRECT)

    def test_zero_means_hold_nothing_mapped(self):
        self.assertEqual(self.reader(shard_handle_limit=0).mode, shards.ShardReader.DIRECT)

    def test_a_meaningless_setting_is_refused_rather_than_guessed_at(self):
        with self.assertRaises(ValueError) as caught:
            LayerLoader(self.root, pool=None, shard_handle_limit="sometimes")
        self.assertIn("auto", str(caught.exception))
        self.assertIn("direct", str(caught.exception))

    def test_the_footprint_and_largest_shard_are_measured_from_the_checkpoint(self):
        reader = self.reader()
        self.assertEqual(reader.checkpoint_bytes(), sum(p.stat().st_size for p in self.shards))
        self.assertEqual(reader.largest_shard_bytes(), self.shard_bytes)

    def test_the_loader_hands_its_settings_to_the_checkpoints_reader(self):
        """The loader owns no file state; it configures the reader everything else shares."""
        loader = self.loader(shard_handle_limit=8, io_workers=2)
        self.assertIs(shards.reader_for(self.root), loader.reader)
        self.assertEqual(loader.reader.handle_limit, 8)
        self.assertEqual(loader.reader.handle_mode, "bounded")
        # Wider than the loader's own pool: the cache prefetches on a pool of its own, and the
        # thread running the forward reads too. A bound sized off io_workers alone is not a bound.
        self.assertGreater(loader.reader.readers, loader.io_workers)

    def test_a_limit_the_readers_cannot_meet_is_reported_as_what_will_be_held(self):
        """The engine's reader count is what decides this, so it is worth pinning down here."""
        loader = self.loader(shard_handle_limit=2, io_workers=2)
        self.assertEqual(loader.reader.handle_limit, loader.reader.readers)
        self.assertIn("cannot share fewer", loader.reader.handle_reason)


# ---- eviction -----------------------------------------------------------------------------------

class TestHandleEviction(LoaderCase):
    """Only the mapped path holds anything open, so these force it on."""

    def test_bounded_mode_never_holds_more_than_its_limit(self):
        reader = self.reader(mapped=True, profile=fake_profile(budget_bytes=self.shard_bytes * 2))
        handles = self.read_all(reader)
        self.assertLessEqual(len(handles), reader._per_thread_limit)
        self.assertEqual(reader.handle_opens, len(self.shards))
        self.assertEqual(reader.handle_evictions, len(self.shards) - reader._per_thread_limit)

    def test_unbounded_mode_holds_them_all(self):
        reader = self.reader(mapped=True, profile=fake_profile(budget_bytes=10 * 1024 ** 3))
        handles = self.read_all(reader)
        self.assertEqual(len(handles), len(self.shards))
        self.assertEqual(reader.handle_evictions, 0)

    def test_the_least_recently_used_shard_is_the_one_dropped(self):
        reader = self.reader(mapped=True, shard_handle_limit=2)
        for index in (0, 1, 0, 2):
            reader.read_tensors(self.root / f"model.layers.{index}.safetensors")
        held = {Path(path).stem for path in reader._local.handles}
        self.assertEqual(held, {"model.layers.0", "model.layers.2"})

    def test_a_repeated_shard_is_not_reopened(self):
        reader = self.reader(mapped=True, shard_handle_limit=2)
        for _ in range(5):
            reader.read_tensors(self.root / "model.layers.0.safetensors")
        self.assertEqual(reader.handle_opens, 1)
        self.assertEqual(reader.handle_evictions, 0)

    def test_an_evicted_handle_is_closed_rather_than_merely_dropped(self):
        """Dropping the reference leaves the mapping alive until the collector runs, which is the
        whole thing this exists to stop."""
        reader = self.reader(mapped=True, shard_handle_limit=1)
        reader.read_tensors(self.root / "model.layers.0.safetensors")
        evicted = next(iter(reader._local.handles.values()))
        reader.read_tensors(self.root / "model.layers.1.safetensors")
        self.assertEqual(reader.handle_evictions, 1)
        with self.assertRaises(Exception):
            evicted.get_slice("model.layers.0.weight")

    def test_release_frees_everything_still_held(self):
        reader = self.reader(mapped=True, shard_handle_limit="unbounded")
        self.read_all(reader)
        self.assertEqual(len(reader._handles), len(self.shards))
        reader.release()
        self.assertEqual(reader._handles, set())

    def test_the_shards_can_be_deleted_after_release(self):
        """A mapped file cannot be unlinked on Windows, so this is the observable proof that the
        mappings are gone rather than merely dereferenced."""
        reader = self.reader(mapped=True, shard_handle_limit="unbounded")
        self.read_all(reader)
        reader.release()
        for shard in self.shards:
            shard.unlink()
        self.assertEqual(list(self.root.glob("*.safetensors")), [])

    def test_closing_the_loader_releases_the_checkpoints_reader(self):
        loader = LayerLoader(self.root, pool=None)
        loader.plan("model.layers.0")
        loader.close()
        self.assertIsNot(shards.reader_for(self.root), loader.reader)
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
        reader = self.reader(mapped=True, shard_handle_limit=1)
        name = "model.layers.0.weight"
        path = self.root / "model.layers.0.safetensors"
        first = reader.read_tensors(path)[name]
        reader.read_tensors(self.root / "model.layers.1.safetensors")   # evicts 0
        again = reader.read_tensors(path)[name]                         # reopens it
        self.assertGreaterEqual(reader.handle_evictions, 1)
        self.assertTrue(torch.equal(first, again))

    def test_planning_reads_no_tensor_data_at_all(self):
        """What lets the engine size a checkpoint far larger than the machine could ever hold."""
        loader = self.loader()
        before = loader.reader.bytes_read
        for index in range(len(self.shards)):
            loader.plan(f"model.layers.{index}")
        self.assertEqual(loader.reader.bytes_read, before)

    def test_stats_report_the_mode_so_a_slow_run_can_be_explained(self):
        stats = self.loader(shard_handle_limit=2).stats()
        for key in ("shard_read_mode", "shard_handle_mode", "shard_handle_limit",
                    "shard_handle_opens", "shard_handle_evictions", "io_workers", "reads",
                    "bytes_read"):
            self.assertIn(key, stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
