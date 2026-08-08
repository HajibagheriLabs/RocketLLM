"""Derivation tests for HardwareProfile, with every probe result mocked.

The probes themselves need real hardware, so what is checked here is the part that has to be right
on hardware nobody testing this owns: that the derived knobs move sensibly across wildly different
machines, and that none of them ever comes out negative, zero where zero is nonsense, or via a
division by something that was never measured.

Four machines, chosen for the corners of the space:

  * a large-VRAM accelerator with fast NVMe
  * a tiny-VRAM card on a small host
  * a machine with no accelerator at all
  * a machine whose weights sit on something very slow

None of these need an accelerator to run, which is the whole point.
"""
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from rocketllm.hw.profile import (DEFAULT_POLICY, Derivation, HardwareProfile, Policy,
                                  SCHEMA_VERSION)

GB = 1024 ** 3


def make_profile(**overrides):
    """A profile whose probe results are supplied rather than measured."""
    base = dict(
        schema_version=SCHEMA_VERSION,
        fingerprint="test",
        probed_at="2026-01-01T00:00:00+0000",
        probe_seconds=1.0,
        backend="cuda",
        device_type="cuda",
        device_name="Test Device",
        device_count=1,
        compute_capability=[8, 6],
        architecture="sm_86",
        driver_version="999.99",
        runtime_version="12.1",
        device_total_bytes=24 * GB,
        device_free_bytes=23 * GB,
        host_total_bytes=64 * GB,
        host_available_bytes=48 * GB,
        cpu_count=16,
        device_memory_bandwidth=900e9,
        host_to_device_pinned_bandwidth=25e9,
        host_to_device_pageable_bandwidth=9e9,
        storage={
            "path": "/weights", "probed_real_shards": True, "synthetic_probe": False,
            "queue_depth_1_bytes_per_s": 2.0e9,
            "by_concurrency": {"1": 2.0e9, "2": 3.4e9, "4": 5.0e9, "8": 5.1e9},
            "best_bytes_per_s": 5.1e9, "saturating_concurrency": 4,
            "page_cache_dropped": True, "page_cache_influence": "reduced",
            "rotational": False, "bytes_read": 512 << 20, "seconds": 1.0, "error": None,
        },
        dtypes={"bf16": True, "fp16": True, "fp8": False, "fp4": False},
        pinned_memory=True,
        async_copy_streams=True,
        triton=True,
        fused_4bit={"torch_int4pack": True, "bitsandbytes": False, "awq": False,
                    "gptq": False, "marlin": False, "any_usable": True},
        allocator={"peak_allocated_bytes": 100 << 20, "peak_reserved_bytes": 110 << 20,
                   "fragmentation_ratio": 0.09, "workspace_bytes": 64 << 20,
                   "expandable_segments": True},
        versions={"python": "3.11.0", "torch": "2.5.1", "platform": "test", "machine": "x86_64",
                  "executable": "python.exe"},
        derived={},
        warnings=[],
        policy=dataclasses.asdict(DEFAULT_POLICY),
    )
    base.update(overrides)
    profile = HardwareProfile(**base)
    profile.derive()
    return profile


def big_vram_machine(**kw):
    return make_profile(**kw)


def tiny_vram_machine(**kw):
    """A 4GB card in a 8GB laptop: the configuration RocketLLM exists for."""
    defaults = dict(
        device_name="Tiny Device", device_total_bytes=4 * GB, device_free_bytes=3 * GB,
        host_total_bytes=8 * GB, host_available_bytes=3 * GB, cpu_count=4,
        compute_capability=[7, 5], architecture="sm_75",
        device_memory_bandwidth=220e9,
        host_to_device_pinned_bandwidth=6e9, host_to_device_pageable_bandwidth=3e9,
        dtypes={"bf16": False, "fp16": True, "fp8": False, "fp4": False},
        fused_4bit={"torch_int4pack": False, "bitsandbytes": False, "awq": False,
                    "gptq": False, "marlin": False, "any_usable": False},
        allocator={"peak_allocated_bytes": 50 << 20, "peak_reserved_bytes": 60 << 20,
                   "fragmentation_ratio": 0.17, "workspace_bytes": 32 << 20,
                   "expandable_segments": False},
        storage={
            "path": "/weights", "probed_real_shards": True, "synthetic_probe": False,
            "queue_depth_1_bytes_per_s": 4.0e8,
            "by_concurrency": {"1": 4.0e8, "2": 6.0e8, "4": 6.2e8},
            "best_bytes_per_s": 6.2e8, "saturating_concurrency": 2,
            "page_cache_dropped": True, "page_cache_influence": "reduced",
            "rotational": False, "bytes_read": 64 << 20, "seconds": 1.0, "error": None,
        },
    )
    defaults.update(kw)
    return make_profile(**defaults)


def no_accelerator_machine(**kw):
    defaults = dict(
        backend="cpu", device_type="cpu", device_name="Some CPU", device_count=8,
        compute_capability=None, architecture=None, driver_version=None, runtime_version=None,
        device_total_bytes=None, device_free_bytes=None,
        host_total_bytes=32 * GB, host_available_bytes=20 * GB, cpu_count=8,
        device_memory_bandwidth=40e9,
        host_to_device_pinned_bandwidth=None, host_to_device_pageable_bandwidth=None,
        dtypes={"bf16": True, "fp16": True, "fp8": False, "fp4": False},
        pinned_memory=False, async_copy_streams=False, triton=False,
        fused_4bit={"torch_int4pack": True, "bitsandbytes": False, "awq": False,
                    "gptq": False, "marlin": False, "any_usable": False},
        allocator={"peak_allocated_bytes": None, "peak_reserved_bytes": None,
                   "fragmentation_ratio": None, "workspace_bytes": None,
                   "expandable_segments": False},
    )
    defaults.update(kw)
    return make_profile(**defaults)


def slow_storage_machine(**kw):
    """Weights on a spinning disk: streaming dominates by orders of magnitude."""
    defaults = dict(
        storage={
            "path": "/mnt/hdd/weights", "probed_real_shards": True, "synthetic_probe": False,
            "queue_depth_1_bytes_per_s": 8.0e7,
            "by_concurrency": {"1": 8.0e7, "2": 8.1e7, "4": 7.9e7},
            "best_bytes_per_s": 8.1e7, "saturating_concurrency": 1,
            "page_cache_dropped": True, "page_cache_influence": "reduced",
            "rotational": True, "bytes_read": 32 << 20, "seconds": 2.0, "error": None,
        },
    )
    defaults.update(kw)
    return make_profile(**defaults)


ALL_MACHINES = {
    "big_vram": big_vram_machine,
    "tiny_vram": tiny_vram_machine,
    "no_accelerator": no_accelerator_machine,
    "slow_storage": slow_storage_machine,
}


class TestNothingIsNonsense(unittest.TestCase):
    """Invariants that must hold on every machine, however strange."""

    def test_no_derived_byte_count_is_negative(self):
        for name, build in ALL_MACHINES.items():
            profile = build()
            for knob, derivation in profile.derived.items():
                if knob.endswith("_bytes"):
                    with self.subTest(machine=name, knob=knob):
                        self.assertIsInstance(derivation.value, int)
                        self.assertGreaterEqual(derivation.value, 0)

    def test_io_workers_is_always_at_least_one(self):
        for name, build in ALL_MACHINES.items():
            with self.subTest(machine=name):
                self.assertGreaterEqual(build().derived["io_workers"].value, 1)

    def test_window_max_is_at_least_one_layer_however_small_the_device(self):
        """Falling to zero would mean "stream nothing", which cannot make progress."""
        for name, build in ALL_MACHINES.items():
            profile = build()
            for layer_bytes in (1, 512 << 20, 8 * GB, 64 * GB, 10 ** 15):
                with self.subTest(machine=name, layer_bytes=layer_bytes):
                    self.assertGreaterEqual(profile.window_max(layer_bytes), 1)

    def test_window_max_survives_a_zero_or_missing_layer_size(self):
        for name, build in ALL_MACHINES.items():
            profile = build()
            for bad in (0, None):
                with self.subTest(machine=name, layer_bytes=bad):
                    self.assertEqual(profile.window_max(bad), 1)

    def test_every_knob_carries_a_formula_and_its_inputs(self):
        for name, build in ALL_MACHINES.items():
            profile = build()
            for knob, derivation in profile.derived.items():
                with self.subTest(machine=name, knob=knob):
                    self.assertTrue(derivation.formula.strip(), f"{knob} has no formula")
                    self.assertIsInstance(derivation.inputs, dict)

    def test_deriving_twice_gives_the_same_answer(self):
        for name, build in ALL_MACHINES.items():
            profile = build()
            first = {k: d.value for k, d in profile.derived.items()}
            profile.derive()
            with self.subTest(machine=name):
                self.assertEqual(first, {k: d.value for k, d in profile.derived.items()})

    def test_a_machine_that_measured_nothing_still_derives(self):
        """Every probe failed. Nothing may divide by zero or invent a number."""
        profile = make_profile(
            device_total_bytes=None, device_free_bytes=None,
            host_total_bytes=None, host_available_bytes=None, cpu_count=None,
            device_memory_bandwidth=None,
            host_to_device_pinned_bandwidth=None, host_to_device_pageable_bandwidth=None,
            storage={"path": None, "best_bytes_per_s": None, "saturating_concurrency": None,
                     "by_concurrency": {}, "error": "not probed", "rotational": None},
            allocator={"fragmentation_ratio": None, "workspace_bytes": None,
                       "expandable_segments": False},
        )
        self.assertEqual(profile.derived["reserve_bytes"].value, 0)
        self.assertEqual(profile.derived["host_cache_bytes"].value, 0)
        self.assertEqual(profile.derived["io_workers"].value, 1)
        self.assertEqual(profile.derived["window_budget_bytes"].value, 0)
        self.assertEqual(profile.window_max(1 << 20), 1)
        # Unmeasurable, so it must say so rather than guess either way.
        self.assertIsNone(profile.derived["speculative_recommended"].value)


class TestReserve(unittest.TestCase):
    def test_reserve_scales_with_the_card(self):
        big = big_vram_machine().derived["reserve_bytes"].value
        tiny = tiny_vram_machine().derived["reserve_bytes"].value
        self.assertGreater(big, tiny)

    def test_reserve_never_exceeds_the_ceiling(self):
        """A pathological fragmentation measurement must not reserve the whole device."""
        profile = make_profile(allocator={"fragmentation_ratio": 0.99,
                                          "workspace_bytes": 100 * GB,
                                          "expandable_segments": True})
        reserve = profile.derived["reserve_bytes"].value
        self.assertLessEqual(reserve, int(profile.device_total_bytes
                                          * DEFAULT_POLICY.reserve_ceiling_fraction))

    def test_reserve_is_zero_without_a_device(self):
        self.assertEqual(no_accelerator_machine().derived["reserve_bytes"].value, 0)

    def test_reserve_leaves_usable_memory_positive(self):
        for name, build in ALL_MACHINES.items():
            profile = build()
            with self.subTest(machine=name):
                self.assertGreaterEqual(profile.derived["usable_device_bytes"].value, 0)


class TestHostCache(unittest.TestCase):
    def test_bigger_free_ram_gives_a_bigger_cache(self):
        roomy = make_profile(host_total_bytes=64 * GB, host_available_bytes=48 * GB)
        cramped = make_profile(host_total_bytes=64 * GB, host_available_bytes=10 * GB)
        self.assertGreater(roomy.derived["host_cache_bytes"].value,
                           cramped.derived["host_cache_bytes"].value)

    def test_a_loaded_machine_computes_to_zero_rather_than_negative(self):
        """Zero is a supported configuration: pure streaming, no host tier. It must not go under."""
        profile = make_profile(host_total_bytes=64 * GB, host_available_bytes=1 * GB)
        self.assertEqual(profile.derived["host_cache_bytes"].value, 0)

    def test_zero_host_cache_is_warned_about(self):
        profile = make_profile(host_total_bytes=64 * GB, host_available_bytes=1 * GB)
        self.assertTrue(any("host cache budget computed to 0" in w for w in profile.warnings))

    def test_cache_never_claims_more_than_available_ram(self):
        for name, build in ALL_MACHINES.items():
            profile = build()
            if profile.host_available_bytes:
                with self.subTest(machine=name):
                    self.assertLess(profile.derived["host_cache_bytes"].value,
                                    profile.host_available_bytes)


class TestIoWorkers(unittest.TestCase):
    def test_workers_follow_the_measured_saturation_point(self):
        self.assertEqual(big_vram_machine().derived["io_workers"].value, 4)
        self.assertEqual(tiny_vram_machine().derived["io_workers"].value, 2)

    def test_slow_serial_storage_gets_one_worker(self):
        """A spinning disk saturates at queue depth 1; more threads only cause seeking."""
        self.assertEqual(slow_storage_machine().derived["io_workers"].value, 1)

    def test_workers_are_capped_by_cpu_count(self):
        profile = make_profile(cpu_count=2, storage=dict(
            big_vram_machine().storage, saturating_concurrency=16))
        self.assertEqual(profile.derived["io_workers"].value, 2)

    def test_unprobed_storage_falls_back_to_one_worker(self):
        profile = make_profile(storage={"path": None, "best_bytes_per_s": None,
                                        "saturating_concurrency": None, "by_concurrency": {},
                                        "error": "not probed", "rotational": None})
        self.assertEqual(profile.derived["io_workers"].value, 1)


class TestDtypes(unittest.TestCase):
    def test_bf16_is_used_where_supported(self):
        self.assertEqual(big_vram_machine().derived["compute_dtype"].value, "bfloat16")

    def test_no_bf16_falls_back_to_fp16_and_warns_about_the_risk(self):
        profile = tiny_vram_machine()
        self.assertEqual(profile.derived["compute_dtype"].value, "float16")
        self.assertTrue(any("overflow" in w for w in profile.warnings),
                        "falling back to fp16 must name the overflow risk")

    def test_neither_dtype_falls_back_to_fp32(self):
        profile = make_profile(dtypes={"bf16": False, "fp16": False, "fp8": False, "fp4": False})
        self.assertEqual(profile.derived["compute_dtype"].value, "float32")

    def test_kv_int4_when_the_device_is_the_smaller_pool(self):
        self.assertEqual(tiny_vram_machine().derived["kv_dtype"].value, "int4")

    def test_kv_follows_compute_when_the_device_is_larger_than_the_host(self):
        profile = make_profile(device_total_bytes=192 * GB, device_free_bytes=190 * GB,
                               host_total_bytes=32 * GB, host_available_bytes=24 * GB)
        self.assertEqual(profile.derived["kv_dtype"].value,
                         profile.derived["compute_dtype"].value)

    def test_kv_choice_does_not_flip_on_small_changes_in_free_ram(self):
        """It keys off total RAM precisely so a browser tab cannot change the answer."""
        busy = tiny_vram_machine(host_available_bytes=1 * GB)
        idle = tiny_vram_machine(host_available_bytes=7 * GB)
        self.assertEqual(busy.derived["kv_dtype"].value, idle.derived["kv_dtype"].value)


class TestQuantPath(unittest.TestCase):
    def test_fused_path_when_a_kernel_is_usable(self):
        self.assertEqual(big_vram_machine().derived["quant_compute_path"].value, "fused_packed")

    def test_dequant_to_scratch_when_no_kernel_is_usable(self):
        profile = tiny_vram_machine()
        self.assertEqual(profile.derived["quant_compute_path"].value, "dequant_to_scratch")
        self.assertTrue(any("dequantized into scratch" in w for w in profile.warnings))

    def test_cpu_never_takes_the_fused_path(self):
        self.assertEqual(no_accelerator_machine().derived["quant_compute_path"].value,
                         "dequant_to_scratch")


class TestSpeculativeRecommendation(unittest.TestCase):
    def test_recommended_when_weights_arrive_far_slower_than_compute(self):
        self.assertIs(slow_storage_machine().derived["speculative_recommended"].value, True)

    def test_not_recommended_when_the_tiers_are_close(self):
        """Nothing to amortize: moving the weights is not what costs."""
        profile = make_profile(
            device_memory_bandwidth=50e9,
            host_to_device_pinned_bandwidth=40e9,
            host_to_device_pageable_bandwidth=40e9,
            storage=dict(big_vram_machine().storage, best_bytes_per_s=45e9))
        self.assertIs(profile.derived["speculative_recommended"].value, False)

    def test_unavailable_rather_than_false_when_it_could_not_be_measured(self):
        profile = make_profile(device_memory_bandwidth=None)
        self.assertIsNone(profile.derived["speculative_recommended"].value)

    def test_a_zero_bandwidth_measurement_does_not_divide_by_zero(self):
        profile = make_profile(
            host_to_device_pinned_bandwidth=0, host_to_device_pageable_bandwidth=0,
            storage=dict(big_vram_machine().storage, best_bytes_per_s=0))
        self.assertIsNone(profile.derived["speculative_recommended"].value)


class TestWarnings(unittest.TestCase):
    def test_slow_storage_is_called_out_with_its_measured_rate(self):
        profile = slow_storage_machine()
        self.assertTrue(any("storage read bandwidth measured" in w for w in profile.warnings))

    def test_rotational_media_is_called_out(self):
        self.assertTrue(any("rotational" in w for w in slow_storage_machine().warnings))

    def test_fast_storage_is_not_called_slow(self):
        self.assertFalse(any("rotational" in w or "storage read bandwidth measured" in w
                             for w in big_vram_machine().warnings))

    def test_missing_pinned_memory_is_reported_once(self):
        warnings = no_accelerator_machine().warnings
        self.assertEqual(sum("pinned host memory" in w for w in warnings), 1)


class TestOverrides(unittest.TestCase):
    def test_an_override_replaces_the_value_and_says_so(self):
        profile = big_vram_machine()
        profile.derive(overrides={"io_workers": 3})
        self.assertEqual(profile.derived["io_workers"].value, 3)
        self.assertEqual(profile.derived["io_workers"].source, "override")

    def test_the_derived_formula_is_kept_next_to_the_override(self):
        """A bug report needs to show what would have been chosen, not just what was forced."""
        profile = big_vram_machine()
        profile.derive(overrides={"reserve_bytes": 1})
        self.assertIn("was:", profile.derived["reserve_bytes"].formula)

    def test_window_fraction_override_changes_the_budget(self):
        profile = big_vram_machine()
        wide = profile.derived["window_budget_bytes"].value
        profile.derive(overrides={"window_fraction": 0.1})
        self.assertLess(profile.derived["window_budget_bytes"].value, wide)

    def test_policy_factors_are_carried_in_the_profile(self):
        strict = Policy(reserve_ceiling_fraction=0.05)
        profile = big_vram_machine()
        profile.derive(policy=strict)
        self.assertLessEqual(profile.derived["reserve_bytes"].value,
                             int(profile.device_total_bytes * 0.05))


class TestSerialisation(unittest.TestCase):
    def test_a_profile_round_trips_through_json(self):
        for name, build in ALL_MACHINES.items():
            profile = build()
            with self.subTest(machine=name):
                restored = HardwareProfile.from_dict(json.loads(json.dumps(profile.to_dict())))
                self.assertEqual(restored.fingerprint, profile.fingerprint)
                self.assertEqual({k: d.value for k, d in restored.derived.items()},
                                 {k: d.value for k, d in profile.derived.items()})

    def test_saving_and_loading_uses_the_fingerprint_as_the_key(self):
        profile = big_vram_machine()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"profile-{profile.fingerprint}.json"
            path.write_text(json.dumps(profile.to_dict(), default=str), encoding="utf-8")
            restored = HardwareProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(restored.device_name, profile.device_name)

    def test_the_fingerprint_moves_when_the_hardware_does(self):
        one = big_vram_machine()
        two = big_vram_machine()
        two.device_total_bytes = 48 * GB
        self.assertNotEqual(one._compute_fingerprint(), two._compute_fingerprint())

    def test_the_fingerprint_is_stable_for_the_same_machine(self):
        self.assertEqual(big_vram_machine()._compute_fingerprint(),
                         big_vram_machine()._compute_fingerprint())

    def test_a_derivation_serialises_with_its_provenance(self):
        entry = Derivation(5, "formula", {"a": 1}, source="override").to_dict()
        self.assertEqual(entry["source"], "override")
        self.assertEqual(entry["formula"], "formula")


class TestDescribe(unittest.TestCase):
    def test_every_machine_renders_without_blowing_up(self):
        for name, build in ALL_MACHINES.items():
            with self.subTest(machine=name):
                text = build().describe()
                self.assertIn("RocketLLM hardware profile", text)
                self.assertIn("DERIVED TUNING KNOBS", text)

    def test_unavailable_measurements_are_named_not_zeroed(self):
        text = no_accelerator_machine().describe()
        self.assertIn("unavailable", text)
        self.assertNotIn("0.00 B/s", text)

    def test_every_knob_appears_with_its_formula(self):
        profile = big_vram_machine()
        text = profile.describe()
        for knob in profile.derived:
            with self.subTest(knob=knob):
                self.assertIn(knob, text)
        self.assertEqual(text.count("formula:"), len(profile.derived))


if __name__ == "__main__":
    unittest.main(verbosity=2)
