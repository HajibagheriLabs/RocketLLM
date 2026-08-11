"""Tests for the int4 KV cache: round-trip bounds, the K/V asymmetry, and the window boundary.

Three things are worth pinning here, and the third is where implementations of this actually break.

**The error bound.** int4 asymmetric quantization has an exact worst case -- half a step, where a step
is the group's range over fifteen. A round trip that exceeds it is not "a bit lossy", it is wrong,
and the failure is quiet: the model still generates, slightly worse. So the bound is asserted against
each group's own measured range rather than against a tolerance someone picked.

**The K/V asymmetry.** K is quantized per channel and V per token, and that is the entire reason this
file exists rather than a call into a generic quantizer. The test for it constructs the situation the
recipe is designed for -- a channel whose values are much larger than its neighbours', which is what
real key projections have -- and requires the per-channel layout to be substantially better on the
*other* channels. Swap the axes and that test fails, which is the point.

**The residual boundary.** The transition from fp16 window to quantized block is where tokens get
duplicated, dropped, or silently reordered, because it is the only place the two storage forms meet.
The tests below walk a cache one token at a time across several flushes and check the invariants that
would catch each of those: the length the cache reports, the length it returns, the exactness of the
window, and the order of what comes out.

Everything here runs on CPU with no accelerator and no model.
"""
import unittest

import torch

from rocketllm.quant.kv_cache import (KV_AUTO, KV_FP16, KV_INT4, KVCacheConfig, QuantizedKVCache,
                                      build_kv_cache, quantize, resolve_kv_cache)

GB = 1024 ** 3


def kv(batch=1, heads=2, tokens=8, dim=64, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, heads, tokens, dim, dtype=torch.float16)


class TestRoundTripBounds(unittest.TestCase):
    """int4 has an exact worst case. Anything beyond it is a bug, not a tolerance."""

    def assert_within_step(self, original, restored, axis, group):
        """Every element must land within half a quantization step of where it started.

        The step is computed from each group's own range, so this is the mathematical bound rather
        than a number chosen to make the test pass. A little slack is allowed for the scale and
        zero point themselves being stored in fp16.
        """
        moved = original.movedim(axis, -1).float()
        length = moved.shape[-1]
        usable = length - (length % group) if length >= group else 0
        self.assertGreater(usable, 0, "the test tensor is shorter than one group")
        head = moved[..., :usable]
        grouped = head.reshape(*head.shape[:-1], usable // group, group)
        spread = grouped.amax(dim=-1) - grouped.amin(dim=-1)
        step = spread / 15.0

        got = restored.movedim(axis, -1).float()[..., :usable]
        error = (got.reshape(*head.shape[:-1], usable // group, group) - grouped).abs().amax(dim=-1)
        # fp16 stores the scale and the zero point, so allow a relative slack on top of half a step.
        self.assertTrue(torch.all(error <= step / 2 + spread * 1e-3 + 1e-4),
                        f"max error {error.max():.5f} exceeds half a step "
                        f"{(step / 2).max():.5f}")

    def test_key_axis_round_trip(self):
        x = kv(tokens=256)
        self.assert_within_step(x, quantize(x, 2, 64).dequantize(), 2, 64)

    def test_value_axis_round_trip(self):
        x = kv(tokens=256)
        self.assert_within_step(x, quantize(x, 3, 64).dequantize(), 3, 64)

    def test_shape_and_dtype_survive(self):
        x = kv(tokens=192, dim=128)
        for axis in (2, 3):
            back = quantize(x, axis, 64).dequantize()
            self.assertEqual(back.shape, x.shape)
            self.assertEqual(back.dtype, x.dtype)

    def test_a_constant_group_is_exact(self):
        """No range means no error. It is also the division-by-zero the scale has to dodge."""
        x = torch.full((1, 1, 64, 64), 0.375, dtype=torch.float16)
        back = quantize(x, 2, 64).dequantize()
        self.assertTrue(torch.equal(back, x))

    def test_an_axis_shorter_than_a_group_still_round_trips(self):
        """head_dim below the group size is a real configuration on small models."""
        x = kv(tokens=8, dim=16)
        back = quantize(x, 3, 64).dequantize()
        self.assertEqual(back.shape, x.shape)
        self.assertLess((back.float() - x.float()).abs().max(), 1.0)

    def test_a_ragged_last_group_does_not_distort_its_scale(self):
        """The padding repeats an existing value, so it cannot widen the group's range.

        Zero padding would: a group of values all near 5.0 would suddenly span 0..5 and every real
        value in it would lose most of its resolution.
        """
        x = torch.full((1, 1, 100, 8), 5.0, dtype=torch.float16)
        back = quantize(x, 2, 64).dequantize()
        self.assertTrue(torch.equal(back, x))

    def test_packing_really_is_four_bits(self):
        """Storing a code per byte would cost what int8 costs and give back half the saving."""
        x = kv(tokens=1024, dim=64)
        packed = quantize(x, 2, 64)
        fp16_bytes = x.numel() * 2
        self.assertLess(packed.nbytes, fp16_bytes / 3,
                        "int4 with group 64 should be comfortably under a third of fp16")
        self.assertEqual(packed.packed.dtype, torch.uint8)


class TestKeyValueAsymmetry(unittest.TestCase):
    """REGRESSION TEST. Swapping the axes is the mistake that makes 4-bit KV look unusable."""

    def channel_outlier(self):
        x = kv(tokens=256, dim=64, seed=3)
        x[..., 7] *= 30.0
        return x

    def test_per_channel_protects_the_other_channels_from_an_outlier(self):
        x = self.channel_outlier()
        ordinary = [d for d in range(x.shape[-1]) if d != 7]

        per_channel = quantize(x, 2, 64).dequantize()   # K: groups along tokens
        per_token = quantize(x, 3, 64).dequantize()     # V layout, wrong for K

        good = (per_channel[..., ordinary].float() - x[..., ordinary].float()).abs().mean()
        bad = (per_token[..., ordinary].float() - x[..., ordinary].float()).abs().mean()
        self.assertLess(good * 3, bad,
                        f"per-channel ({good:.5f}) should be far better than per-token ({bad:.5f}) "
                        f"on the channels that do not hold the outlier; if this fails the key and "
                        f"value axes have probably been swapped")

    def test_the_cache_uses_the_split_by_default(self):
        config = KVCacheConfig()
        self.assertEqual(config.key_axis, 2, "K must group along tokens, giving each channel a scale")
        self.assertEqual(config.value_axis, 3, "V must group along channels, giving each token one")


class TestResidualWindowBoundary(unittest.TestCase):
    """Where the fp16 window meets the quantized blocks, and where these break."""

    def build(self, group=64, residual=128):
        return QuantizedKVCache(KVCacheConfig(group_size=group, residual_length=residual))

    def feed(self, cache, tokens, dim=64, layer=0, start=0):
        keys = torch.arange(start, start + tokens, dtype=torch.float16)
        keys = keys.view(1, 1, tokens, 1).expand(1, 2, tokens, dim).contiguous()
        return cache.update(keys, keys.clone(), layer)

    def test_nothing_is_quantized_before_the_window_fills(self):
        cache = self.build()
        out, _ = self.feed(cache, 128)
        self.assertEqual(cache.report()["quantized_tokens"], 0)
        self.assertEqual(cache.report()["flushes"], 0)
        self.assertEqual(out.shape[-2], 128)

    def test_the_window_never_drops_below_its_length(self):
        """The most recent tokens are the ones worth keeping exact; the window must not shrink."""
        cache = self.build()
        for step in range(400):
            self.feed(cache, 1, start=step)
            self.assertGreaterEqual(cache.report()["residual_tokens"], 0)
            if cache.report()["flushes"]:
                self.assertGreaterEqual(cache.report()["residual_tokens"], 128)

    def test_flushes_happen_in_whole_groups(self):
        """A partial group would need a scale computed from fewer tokens than every other group."""
        cache = self.build()
        for step in range(400):
            self.feed(cache, 1, start=step)
            self.assertEqual(cache.report()["quantized_tokens"] % 64, 0)

    def test_length_is_conserved_across_the_boundary(self):
        cache = self.build()
        for step in range(400):
            out, _ = self.feed(cache, 1, start=step)
            self.assertEqual(cache.get_seq_length(0), step + 1)
            self.assertEqual(out.shape[-2], step + 1,
                             "the cache returned a different number of tokens than it holds")

    def test_the_window_is_bit_exact(self):
        """Whatever is still in fp16 must come back untouched, not merely close."""
        cache = self.build()
        original = []
        for step in range(300):
            keys = torch.randn(1, 2, 1, 64, dtype=torch.float16)
            original.append(keys)
            out, _ = cache.update(keys, keys.clone(), 0)
        residual = cache.report()["residual_tokens"]
        expected = torch.cat(original[-residual:], dim=-2)
        self.assertTrue(torch.equal(out[..., -residual:, :], expected),
                        "the residual window is not returned exactly")

    def test_order_survives_a_flush(self):
        """Tokens must come back in the order they went in, across the block/window seam."""
        cache = self.build()
        for step in range(300):
            self.feed(cache, 1, start=step)
        out, _ = self.feed(cache, 1, start=300)
        recovered = out[0, 0, :, 0].float()
        # The ramp is monotonic, so any reordering or duplication shows up immediately. Quantized
        # values are approximate, so compare against the ramp with the step's tolerance.
        self.assertEqual(recovered.shape[0], 301)
        drift = (recovered - torch.arange(301, dtype=torch.float32)).abs()
        self.assertLess(drift.max(), 8.0, "tokens came back out of order or duplicated")
        # Non-decreasing rather than increasing: sixty-four consecutive ramp values share sixteen
        # levels, so neighbours legitimately tie. A step *down* would mean reordering.
        self.assertTrue(torch.all(recovered.diff() >= 0), "the sequence went backwards")
        # The window is exact, so within it the ramp must still be strictly increasing.
        residual = cache.report()["residual_tokens"]
        self.assertTrue(torch.all(recovered[-residual:].diff() > 0),
                        "the fp16 window lost its ordering")

    def test_a_single_long_prefill_matches_token_by_token_feeding(self):
        """A prompt arrives at once and decode arrives one at a time; both must land the same way."""
        bulk = self.build()
        self.feed(bulk, 300)
        drip = self.build()
        for step in range(300):
            self.feed(drip, 1, start=step)
        self.assertEqual(bulk.get_seq_length(0), drip.get_seq_length(0))
        self.assertEqual(bulk.report()["quantized_tokens"] % 64, 0)
        self.assertEqual(drip.report()["quantized_tokens"] % 64, 0)

    def test_a_window_smaller_than_a_group_is_refused(self):
        """It could never complete a group, so nothing would ever be quantized."""
        with self.assertRaises(ValueError):
            KVCacheConfig(group_size=64, residual_length=32)

    def test_several_layers_stay_independent(self):
        cache = self.build()
        self.feed(cache, 200, layer=0)
        self.feed(cache, 200, layer=1)
        self.assertEqual(cache.get_seq_length(0), 200)
        self.assertEqual(cache.get_seq_length(1), 200)
        self.assertEqual(len(cache), 2)

    def test_memory_falls_as_the_context_grows(self):
        """The saving is asymptotic: the fixed window dominates a short context."""
        ratios = {}
        for tokens in (256, 2048):
            cache = self.build()
            keys = torch.randn(1, 4, tokens, 64, dtype=torch.float16)
            cache.update(keys, keys.clone(), 0)
            ratios[tokens] = (keys.numel() * 2 * 2) / cache.nbytes()
        self.assertGreater(ratios[2048], ratios[256])
        self.assertGreater(ratios[2048], 2.5)


class TestChoosing(unittest.TestCase):
    def test_a_model_that_fits_keeps_the_context_exact(self):
        choice, reason = resolve_kv_cache(KV_AUTO, weight_bytes=4 * GB, device_bytes=20 * GB,
                                          headroom=0.15)
        self.assertEqual(choice, KV_FP16)
        self.assertIn("fit", reason)

    def test_a_model_that_does_not_fit_buys_context(self):
        choice, reason = resolve_kv_cache(KV_AUTO, weight_bytes=40 * GB, device_bytes=20 * GB,
                                          headroom=0.15)
        self.assertEqual(choice, KV_INT4)

    def test_headroom_is_required_not_just_a_bare_fit(self):
        """Fitting with nothing left over is not fitting: the context has to go somewhere too."""
        choice, _ = resolve_kv_cache(KV_AUTO, weight_bytes=20 * GB, device_bytes=21 * GB,
                                     headroom=0.15)
        self.assertEqual(choice, KV_INT4)

    def test_a_bigger_card_does_not_downgrade_a_model_that_still_does_not_fit(self):
        """The original concern, kept: more memory must not cost context on a model that needs it."""
        for total in (24, 48, 80, 192):
            choice, _ = resolve_kv_cache(KV_AUTO, weight_bytes=400 * GB,
                                         device_bytes=total * GB, headroom=0.15)
            with self.subTest(device_gb=total):
                self.assertEqual(choice, KV_INT4)

    def test_an_explicit_setting_is_never_second_guessed(self):
        for setting in (KV_FP16, KV_INT4):
            choice, reason = resolve_kv_cache(setting, weight_bytes=40 * GB, device_bytes=1 * GB)
            self.assertEqual(choice, setting)
            self.assertIn("explicit", reason)

    def test_nothing_measured_keeps_the_context_exact(self):
        choice, _ = resolve_kv_cache(KV_AUTO, weight_bytes=None, device_bytes=None)
        self.assertEqual(choice, KV_FP16)

    def test_an_unknown_setting_is_refused(self):
        with self.assertRaises(ValueError):
            resolve_kv_cache("int8")

    def test_fp16_builds_no_cache_at_all(self):
        """The unquantized path has to stay the stock transformers one, byte for byte."""
        self.assertIsNone(build_kv_cache(KV_FP16))

    def test_a_missing_backend_degrades_instead_of_raising(self):
        """hqq and quanto are optional; neither may be required to run a model."""
        cache = build_kv_cache("hqq", KVCacheConfig())
        self.assertIsInstance(cache, QuantizedKVCache)


class TestFreedMemoryReachesThePinBudget(unittest.TestCase):
    """The other half of quantizing the context: the memory it frees has to become resident weights.

    Freeing bytes is worth nothing on its own. The chain is: the context shrinks, the device budget
    measures more free memory, the cache is resized, the pin budget grows, and more weights stay
    resident. These call the engine's own callback rather than re-implementing it, because what is
    being checked is that the wiring exists -- and until this change it did not: VramBudget was
    written, tested, and had no caller anywhere in the engine.

    Note that end-to-end this is damped: the budget only republishes once a change has persisted and
    exceeded a hysteresis band derived from total device memory, which on a large card is far bigger
    than a KV cache's worth of bytes. What is asserted here is the mechanism, not that any given
    context length will trip it.
    """

    def build(self, capacity):
        from rocketllm.base import RocketModel
        from rocketllm.memory.cache import TieredWeightCache

        cache = TieredWeightCache(fetch=lambda key: f"payload:{key}",
                                  sizer=lambda key: 10 * 1024 * 1024,
                                  device_bytes=capacity, host_bytes=0, window=2)

        class Harness(RocketModel):
            # The engine's real callback, without the loading machinery around it.
            def __init__(self):
                pass

            def _pin_candidates(self):
                from rocketllm.memory import CLASS_ALWAYS, PinCandidate
                return [PinCandidate(key=(i, "dense"), packed_bytes=10 * 1024 * 1024,
                                     priority=CLASS_ALWAYS, accesses_per_token=1.0)
                        for i in range(16)]

        harness = Harness()
        harness.cache = cache
        harness.pin_policy = "auto"
        harness._pin_budget = 0
        harness._window_share = 0.5
        return harness

    def sample(self, estimated=False):
        import types
        return types.SimpleNamespace(estimated=estimated)

    def test_more_free_memory_pins_more_weights(self):
        harness = self.build(capacity=0)
        harness._budget_changed(0, 40 * 1024 * 1024, self.sample())
        few = len(harness.cache.pinned)
        harness._budget_changed(40 * 1024 * 1024, 200 * 1024 * 1024, self.sample())
        many = len(harness.cache.pinned)

        self.assertGreater(many, few,
                           "memory freed by the context did not turn into resident weights")
        self.assertGreater(harness._pin_budget, 0)

    def test_the_capacity_counts_what_the_cache_already_holds(self):
        """The budget measures what is FREE, and what the cache holds is not free.

        Sizing the cache to the free figure alone would tell a cache holding a gigabyte that its
        budget is the few hundred megabytes left over, and it would evict most of itself to fit.
        """
        harness = self.build(capacity=100 * 1024 * 1024)
        harness.cache.acquire((0, "dense"))
        harness.cache.release((0, "dense"))
        held = harness.cache.device.bytes
        self.assertGreater(held, 0)
        self.assertEqual(harness._capacity_for(50 * 1024 * 1024), 50 * 1024 * 1024 + held)

    def test_a_reading_the_backend_could_not_verify_is_not_acted_on(self):
        """CPU reports host RAM as free memory; following it would replace a derived budget."""
        harness = self.build(capacity=1024)
        harness._budget_changed(1024, 8 * 1024 ** 3, self.sample(estimated=True))
        self.assertEqual(harness.cache.device.capacity, 1024)
        self.assertEqual(harness._pin_budget, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
