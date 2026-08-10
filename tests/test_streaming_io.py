"""Tests for the loader, the copy stream and the host-buffer lifetime guard.

The guard is the reason this file exists. An asynchronous transfer keeps reading its host buffer
after the call that issued it has returned, so releasing that buffer early does not raise -- it lets
the next layer write over bytes still in flight, and the model goes on generating with a few wrong
weights. There is no exception to catch and no assertion that fires in production, which is why the
release path is pinned here with a fake event that can be told not to have fired yet.

No accelerator: the copy stream degrades to the synchronous stand-in on CPU, which is the fallback
path a Tier-4 runner takes anyway, and the async ordering is exercised with fakes.
"""
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from rocketllm.hw.caps import CpuCaps
from rocketllm.streaming.loader import ALIGNMENT, LayerLoader, torch_dtype_of
from rocketllm.streaming.staging import BufferLease, HostStagingPool
from rocketllm.streaming.transfer import TransferHandle, WeightTransfer

MB = 1024 ** 2


def make_shard(directory, layer_name, tensors):
    save_file(tensors, str(Path(directory) / f"{layer_name}.safetensors"))


class LoaderTestCase(unittest.TestCase):
    LAYER = "model.layers.0"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.tensors = {
            f"{self.LAYER}.self_attn.q_proj.weight": torch.randn(8, 16, dtype=torch.float32),
            f"{self.LAYER}.mlp.up_proj.weight": torch.randn(32, 8, dtype=torch.float16),
            f"{self.LAYER}.mlp.down_proj.qweight": torch.randint(0, 255, (4, 6),
                                                                 dtype=torch.uint8),
            f"{self.LAYER}.experts.gate_up_proj": torch.arange(60, dtype=torch.int32).reshape(5, 12),
        }
        make_shard(self.dir, self.LAYER, self.tensors)
        self.caps = CpuCaps(torch.device("cpu"))
        self.pool = HostStagingPool(self.caps, 64 * MB)
        self.addCleanup(self.tmp.cleanup)

    def loader(self, **kwargs):
        loader = LayerLoader(self.dir, self.pool, **kwargs)
        self.addCleanup(loader.close)
        return loader


class TestLayout(LoaderTestCase):
    def test_the_plan_reads_only_the_header(self):
        """Sizing a layer must not cost reading it, or the cache cannot decide anything cheaply."""
        loader = self.loader()
        layout = loader.plan(self.LAYER)
        self.assertEqual(len(layout.placements), len(self.tensors))
        self.assertEqual(loader.bytes_read, 0)
        self.assertGreater(layout.total_bytes, 0)

    def test_every_placement_is_aligned_for_its_dtype(self):
        """A slice of a uint8 buffer cannot be reinterpreted at an arbitrary byte offset."""
        layout = self.loader().plan(self.LAYER)
        for placement in layout.placements:
            with self.subTest(tensor=placement.name):
                self.assertEqual(placement.offset % ALIGNMENT, 0)

    def test_placements_do_not_overlap(self):
        layout = self.loader().plan(self.LAYER)
        spans = sorted((p.offset, p.offset + p.nbytes) for p in layout.placements)
        for (_, end), (start, _) in zip(spans, spans[1:]):
            self.assertLessEqual(end, start, "two tensors were given overlapping buffer slices")

    def test_the_dtype_codes_map_to_what_the_file_holds(self):
        for code, expected in (("F32", torch.float32), ("F16", torch.float16),
                               ("BF16", torch.bfloat16), ("U8", torch.uint8),
                               ("I32", torch.int32)):
            with self.subTest(code=code):
                self.assertIs(torch_dtype_of(code), expected)

    def test_an_unrepresentable_dtype_says_so(self):
        with self.assertRaises(ValueError):
            torch_dtype_of("F4_NONSENSE")


class TestReading(LoaderTestCase):
    def assert_roundtrip(self, loader):
        staged = loader.load(self.LAYER)
        try:
            views = staged.views()
            self.assertEqual(set(views), set(self.tensors))
            for name, expected in self.tensors.items():
                self.assertTrue(torch.equal(views[name], expected), f"{name} came back wrong")
        finally:
            staged.release()

    def test_a_layer_round_trips_through_one_contiguous_buffer(self):
        self.assert_roundtrip(self.loader(io_workers=1))

    def test_parallel_workers_produce_identical_bytes(self):
        """Workers write disjoint slices, so concurrency must not change a single byte."""
        self.assert_roundtrip(self.loader(io_workers=4))

    def test_the_whole_layer_lands_in_a_single_buffer(self):
        loader = self.loader()
        staged = loader.load(self.LAYER)
        try:
            buffer = staged.buffer
            for view in staged.views().values():
                self.assertEqual(view.untyped_storage().data_ptr(),
                                 buffer.untyped_storage().data_ptr(),
                                 "a tensor was not a view into the shared staging buffer")
        finally:
            staged.release()

    def test_reading_a_subset_reads_only_that_subset(self):
        loader = self.loader()
        keys = [f"{self.LAYER}.mlp.up_proj.weight"]
        staged = loader.load(self.LAYER, keys=keys)
        try:
            self.assertEqual(list(staged.views()), keys)
            self.assertEqual(loader.bytes_read, self.tensors[keys[0]].numel() * 2)
        finally:
            staged.release()

    def test_a_row_slice_costs_only_that_experts_bytes(self):
        """One expert out of a fused 3D tensor, which is what makes per-expert streaming cheap."""
        loader = self.loader()
        key = f"{self.LAYER}.experts.gate_up_proj"
        staged = loader.load(self.LAYER, keys=[key], rows={key: (2, 3)})
        try:
            view = staged.views()[key]
            self.assertEqual(tuple(view.shape), (1, 12))
            self.assertTrue(torch.equal(view, self.tensors[key][2:3]))
            # A twelfth of the tensor: one row of five, four bytes an element.
            self.assertEqual(loader.bytes_read, 12 * 4)
        finally:
            staged.release()

    def test_io_workers_comes_from_the_profile(self):
        class FakeDerivation:
            value = 6

        class FakeProfile:
            derived = {"io_workers": FakeDerivation()}

        self.assertEqual(self.loader(profile=FakeProfile()).io_workers, 6)

    def test_a_failed_read_does_not_leak_the_lease(self):
        loader = self.loader()
        before = self.pool.leased
        with self.assertRaises(Exception):
            loader.load(self.LAYER, keys=["no.such.tensor"])
        self.assertEqual(self.pool.leased, before, "a failed load left a buffer checked out")


class FakeEvent:
    """An event that only reports completion when told to."""

    def __init__(self, complete=False):
        self.complete = complete
        self.waits = 0
        self.syncs = 0

    def query(self):
        return self.complete

    def wait(self, stream=None):
        self.waits += 1

    def synchronize(self):
        self.syncs += 1
        self.complete = True

    def record(self, stream=None):
        pass


class TestBufferLifetimeGuard(unittest.TestCase):
    """The release path. Getting this wrong corrupts weights and raises nothing."""

    def setUp(self):
        self.caps = CpuCaps(torch.device("cpu"))
        self.pool = HostStagingPool(self.caps, 8 * MB)
        self.transfer = WeightTransfer(self.caps, pool=self.pool)

    def handle(self, complete=False):
        lease = self.pool.lease(1024, torch.uint8)
        event = FakeEvent(complete=complete)
        return TransferHandle(torch.zeros(4), event, lease, self.transfer), lease, event

    def test_an_incomplete_transfer_blocks_rather_than_releasing_its_buffer(self):
        handle, lease, event = self.handle(complete=False)
        handle._release_source()
        self.assertEqual(event.syncs, 1, "the buffer was released without waiting for the copy")
        self.assertTrue(lease.released)

    def test_a_complete_transfer_releases_without_blocking(self):
        handle, lease, event = self.handle(complete=True)
        handle._release_source()
        self.assertEqual(event.syncs, 0)
        self.assertTrue(lease.released)

    def test_reclaim_leaves_an_unfinished_transfers_buffer_alone(self):
        handle, lease, event = self.handle(complete=False)
        self.assertFalse(handle.reclaim_if_complete())
        self.assertFalse(lease.released, "a buffer still being read was handed back")
        self.assertEqual(event.syncs, 0, "reclaim must never block")

    def test_reclaim_collects_a_finished_transfer(self):
        handle, lease, event = self.handle(complete=True)
        self.assertTrue(handle.reclaim_if_complete())
        self.assertTrue(lease.released)

    def test_releasing_twice_is_harmless(self):
        handle, lease, _ = self.handle(complete=True)
        handle._release_source()
        self.assertFalse(handle._release_source())

    def test_a_leased_buffer_is_never_handed_out_again(self):
        first = self.pool.lease(4096, torch.uint8)
        second = self.pool.lease(4096, torch.uint8)
        self.assertIsNot(first.view, second.view)
        self.assertNotEqual(first.view.data_ptr(), second.view.data_ptr(),
                            "the pool handed out a buffer that was still checked out")

    def test_a_released_buffer_is_reused(self):
        first = self.pool.lease(4096, torch.uint8)
        pointer = first.view.data_ptr()
        first.release()
        second = self.pool.lease(4096, torch.uint8)
        self.assertEqual(second.view.data_ptr(), pointer)

    def test_an_exhausted_pool_hands_out_a_private_buffer_rather_than_recycling(self):
        tiny = HostStagingPool(self.caps, 0)
        leases = [tiny.lease(1024, torch.uint8) for _ in range(3)]
        pointers = {lease.view.data_ptr() for lease in leases}
        self.assertEqual(len(pointers), 3, "a zero-budget pool aliased two live buffers")

    def test_clear_keeps_a_buffer_that_is_still_checked_out(self):
        lease = self.pool.lease(4096, torch.uint8)
        self.pool.clear()
        self.assertFalse(lease.released)
        self.assertIsNotNone(lease.view)

    def test_drain_releases_everything_outstanding(self):
        handles = [self.handle(complete=False)[0] for _ in range(3)]
        for handle in handles:
            with self.transfer._lock:
                self.transfer._in_flight.append(handle)
        self.assertEqual(self.transfer.drain(), 3)
        self.assertEqual(self.pool.leased, 0)


class TestSynchronousFallback(unittest.TestCase):
    """CPU has no copy streams, so this is the path a Tier-4 runner actually takes."""

    def setUp(self):
        self.caps = CpuCaps(torch.device("cpu"))
        self.pool = HostStagingPool(self.caps, 8 * MB)
        self.transfer = WeightTransfer(self.caps, pool=self.pool)

    def test_the_backend_without_streams_selects_the_synchronous_path(self):
        self.assertFalse(self.transfer.is_async)

    def test_the_synchronous_path_releases_its_buffer_immediately_and_correctly(self):
        lease = self.pool.lease(64, torch.uint8)
        lease.view.copy_(torch.arange(64, dtype=torch.uint8))
        handle = self.transfer.send_buffer(lease.view, lease)
        self.assertTrue(lease.released, "the sync path finished the copy but kept the buffer")
        self.assertTrue(torch.equal(handle.resolve().cpu(),
                                    torch.arange(64, dtype=torch.uint8)))

    def test_resolving_the_synchronous_handle_is_a_no_op(self):
        lease = self.pool.lease(16, torch.uint8)
        handle = self.transfer.send_buffer(lease.view, lease)
        self.assertIs(handle.resolve(), handle.resolve())

    def test_bytes_survive_the_round_trip_through_the_transfer(self):
        payload = torch.randint(0, 255, (4096,), dtype=torch.uint8)
        lease = self.pool.lease(payload.numel(), torch.uint8)
        lease.view.copy_(payload)
        result = self.transfer.send_buffer(lease.view, lease).resolve()
        self.assertTrue(torch.equal(result.cpu(), payload))

    def test_stats_report_what_moved(self):
        lease = self.pool.lease(1024, torch.uint8)
        self.transfer.send_buffer(lease.view, lease)
        stats = self.transfer.stats()
        self.assertEqual(stats["transfers"], 1)
        self.assertEqual(stats["bytes_sent"], 1024)
        self.assertFalse(stats["async"])


class TestEndToEndStaging(LoaderTestCase):
    def test_a_loaded_layer_transfers_and_matches_the_checkpoint(self):
        loader = self.loader(io_workers=2)
        transfer = WeightTransfer(self.caps, pool=self.pool)
        staged = loader.load(self.LAYER)
        layout = staged.layout
        handle = transfer.send(staged)
        device_buffer = handle.resolve()
        for placement in layout.placements:
            with self.subTest(tensor=placement.name):
                self.assertTrue(torch.equal(placement.view_into(device_buffer).cpu(),
                                            self.tensors[placement.name]))
        transfer.drain()
        self.assertEqual(self.pool.leased, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
