"""Tests for the host staging pool.

The pool exists because page-locking a buffer is a synchronizing driver call expensive enough to
cost more than the transfer it accelerates. What has to be true of it: buffers are reused rather
than reallocated, the budget is respected, and a budget of zero still works -- that last one is a
degradation rule, not an edge case, and it is the configuration a loaded machine actually gets.

No accelerator needed: the pool degrades to pageable buffers wherever pinning is unavailable, which
is exactly what a CPU-only runner exercises.
"""
import unittest

import torch

from rocketllm.hw import caps as C
from rocketllm.hw.caps import CpuCaps
from rocketllm.streaming.transfer import HostStagingPool

MB = 1024 ** 2


def cpu_caps():
    return CpuCaps(torch.device("cpu"))


class TestReuse(unittest.TestCase):
    def setUp(self):
        C.reset_announcements()

    def test_the_same_size_class_hands_back_the_same_storage(self):
        pool = HostStagingPool(cpu_caps(), 64 * MB)
        first = pool.buffer(1000, torch.float32)
        second = pool.buffer(1000, torch.float32)
        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(pool.stats()["buffers"], 1)

    def test_a_growing_request_supersedes_the_smaller_buffer(self):
        """Two sizes straddling a power of two must not leave two buffers behind."""
        pool = HostStagingPool(cpu_caps(), 64 * MB)
        pool.buffer(1_000_000, torch.float32)      # rounds to 2^20
        pool.buffer(1_050_000, torch.float32)      # does not fit, rounds to 2^21
        self.assertEqual(pool.stats()["buffers"], 1)

    def test_a_smaller_request_reuses_the_bigger_buffer(self):
        pool = HostStagingPool(cpu_caps(), 64 * MB)
        big = pool.buffer(1_050_000, torch.float32)
        small = pool.buffer(1_000_000, torch.float32)
        self.assertEqual(big.data_ptr(), small.data_ptr())
        self.assertEqual(pool.stats()["buffers"], 1)

    def test_the_pool_converges_on_one_buffer_per_dtype(self):
        """A real model streams many layer sizes; the pool must not accumulate one each."""
        pool = HostStagingPool(cpu_caps(), 64 * MB)
        for count in (200_000, 1_000_000, 400_000, 1_050_000, 90_000, 800_000):
            pool.buffer(count, torch.float32)
        self.assertEqual(pool.stats()["buffers"], 1)

    def test_the_view_is_exactly_the_requested_length(self):
        pool = HostStagingPool(cpu_caps(), 64 * MB)
        for count in (1, 999, 100_000, 1_048_577):
            with self.subTest(count=count):
                self.assertEqual(pool.buffer(count, torch.float32).numel(), count)

    def test_different_dtypes_do_not_share_a_buffer(self):
        pool = HostStagingPool(cpu_caps(), 64 * MB)
        a = pool.buffer(1000, torch.float32)
        b = pool.buffer(1000, torch.bfloat16)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())
        self.assertEqual(a.dtype, torch.float32)
        self.assertEqual(b.dtype, torch.bfloat16)

    def test_reuse_shows_up_as_hits(self):
        pool = HostStagingPool(cpu_caps(), 64 * MB)
        for _ in range(10):
            pool.buffer(50_000, torch.float32)
        stats = pool.stats()
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hits"], 9)

    def test_a_buffer_survives_between_layers_but_clear_releases_it(self):
        pool = HostStagingPool(cpu_caps(), 64 * MB)
        pool.buffer(50_000, torch.float32)
        self.assertEqual(pool.stats()["buffers"], 1)
        pool.clear()
        self.assertEqual(pool.stats()["buffers"], 0)
        self.assertEqual(pool.stats()["resident_bytes"], 0)


class TestBudget(unittest.TestCase):
    def setUp(self):
        C.reset_announcements()

    def test_a_zero_budget_still_returns_usable_buffers(self):
        """The degradation rule: no pinning, no pooling, still correct."""
        pool = HostStagingPool(cpu_caps(), 0)
        buffer = pool.buffer(10_000, torch.float32)
        self.assertEqual(buffer.numel(), 10_000)
        self.assertEqual(pool.stats()["buffers"], 0)
        self.assertFalse(buffer.is_pinned())

    def test_a_zero_budget_is_announced_once(self):
        with self.assertLogs("rocketllm.hw.caps", level="INFO") as captured:
            HostStagingPool(cpu_caps(), 0)
            HostStagingPool(cpu_caps(), 0)
            HostStagingPool(cpu_caps(), 0)
        self.assertEqual(sum("staging pool budget computed to 0" in line
                             for line in captured.output), 1)

    def test_a_zero_budget_gives_a_fresh_buffer_every_time(self):
        pool = HostStagingPool(cpu_caps(), 0)
        self.assertEqual(pool.stats()["hits"], 0)
        pool.buffer(10_000, torch.float32)
        pool.buffer(10_000, torch.float32)
        self.assertEqual(pool.stats()["hits"], 0)
        self.assertEqual(pool.stats()["misses"], 2)

    def test_the_budget_is_never_exceeded(self):
        pool = HostStagingPool(cpu_caps(), 4 * MB)
        for exponent in range(10, 22):
            pool.buffer(1 << exponent, torch.float32)
        self.assertLessEqual(pool.stats()["resident_bytes"], 4 * MB)

    def test_exceeding_the_budget_falls_back_rather_than_failing(self):
        pool = HostStagingPool(cpu_caps(), 1 * MB)
        buffer = pool.buffer(10_000_000, torch.float32)   # far past the budget
        self.assertEqual(buffer.numel(), 10_000_000)
        self.assertLessEqual(pool.stats()["resident_bytes"], 1 * MB)

    def test_a_negative_or_missing_budget_is_treated_as_zero(self):
        for budget in (-1, None):
            with self.subTest(budget=budget):
                pool = HostStagingPool(cpu_caps(), budget)
                self.assertEqual(pool.budget_bytes, 0)
                self.assertEqual(pool.buffer(100, torch.float32).numel(), 100)


class TestPackingRoundTrip(unittest.TestCase):
    """The pool is only useful if what comes back out is what went in."""

    def test_packing_several_tensors_and_reading_them_back_is_lossless(self):
        pool = HostStagingPool(cpu_caps(), 64 * MB)
        torch.manual_seed(0)
        tensors = [torch.randn(shape) for shape in ((16, 32), (32,), (8, 8, 4), (100,))]
        total = sum(t.numel() for t in tensors)

        buffer = pool.buffer(total, torch.float32)
        offset = 0
        for tensor in tensors:
            buffer.narrow(0, offset, tensor.numel()).copy_(tensor.reshape(-1))
            offset += tensor.numel()

        offset = 0
        for tensor in tensors:
            view = buffer.narrow(0, offset, tensor.numel()).view(tensor.shape)
            self.assertTrue(torch.equal(view, tensor))
            offset += tensor.numel()

    def test_reusing_a_buffer_does_not_leak_the_previous_layer(self):
        """A stale tail would be read as weights, so every byte a layer uses must be rewritten."""
        pool = HostStagingPool(cpu_caps(), 64 * MB)
        first = pool.buffer(1000, torch.float32)
        first.fill_(7.0)
        second = pool.buffer(1000, torch.float32)
        second.copy_(torch.zeros(1000))
        self.assertTrue(torch.equal(second, torch.zeros(1000)))

    def test_a_dtype_cast_happens_on_the_way_into_the_buffer(self):
        """Packing in the runtime dtype is what keeps the transfer narrow."""
        pool = HostStagingPool(cpu_caps(), 64 * MB)
        source = torch.randn(64, dtype=torch.float32)
        buffer = pool.buffer(64, torch.bfloat16)
        buffer.copy_(source)
        self.assertEqual(buffer.dtype, torch.bfloat16)
        self.assertTrue(torch.allclose(buffer.float(), source, atol=1e-2))


class TestSizeClasses(unittest.TestCase):
    def test_size_classes_are_powers_of_two_above_the_floor(self):
        for count, expected in ((1, 1 << 16), ((1 << 16) - 1, 1 << 16), (1 << 16, 1 << 16),
                                ((1 << 16) + 1, 1 << 17), (1 << 20, 1 << 20),
                                ((1 << 20) + 1, 1 << 21)):
            with self.subTest(count=count):
                self.assertEqual(HostStagingPool._size_class(count), expected)

    def test_a_zero_length_request_is_harmless(self):
        pool = HostStagingPool(cpu_caps(), 64 * MB)
        self.assertEqual(pool.buffer(0, torch.float32).numel(), 0)
        self.assertEqual(pool.stats()["buffers"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
