"""`rocketllm doctor`, checked against machines that do not exist here.

The doctor is the thing a bug reporter is asked to paste, which makes its failure mode particular:
if it is wrong, it is wrong in a bug report, and someone spends a day chasing a number nobody
measured. So what is pinned here is not that it prints something, but that what it prints is either
measured or explicitly absent -- never interpolated, never rounded up out of a missing value.

Every machine below is mocked, including ones with no accelerator and ones whose storage could not
be probed at all, because those are the machines the author cannot reach and they are exactly where
an invented number would go unnoticed.
"""
import io
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from rocketllm.hw import caps, doctor
from rocketllm.hw.doctor import (OPTIONAL_PACKAGES, capability_rows, checkpoint_bytes,
                                 model_bytes_from, per_token_projection, report, storage_health)

# The mocked machines live next door, and this file is run both through pytest (where tests/ is a
# package) and directly from a checkout, so the directory goes on the path either way.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_hw_profile import GB, big_vram_machine, make_profile, no_accelerator_machine  # noqa: E402

CPU = caps.CpuCaps(torch.device("cpu"))


def storage(**overrides):
    base = dict(path="/weights", probed_real_shards=True, synthetic_probe=False,
                queue_depth_1_bytes_per_s=1.0e9, by_concurrency={"1": 1.0e9, "4": 2.0e9},
                best_bytes_per_s=2.0e9, saturating_concurrency=4, page_cache_dropped=True,
                page_cache_influence="reduced", rotational=False, bytes_read=64 << 20,
                seconds=1.0, error=None)
    base.update(overrides)
    return base


class TestThePlacement(unittest.TestCase):
    """Where the bytes end up decides everything else, so it is checked before the arithmetic."""

    def test_a_model_that_fits_is_served_entirely_from_the_device(self):
        profile = big_vram_machine()
        pin_budget = (profile.derived["usable_device_bytes"].value
                      - profile.derived["window_budget_bytes"].value)
        projection = per_token_projection(profile, pin_budget // 2)
        tiers = {tier["tier"]: tier for tier in projection["tiers"]}
        self.assertEqual(tiers["host"]["bytes"], 0)
        self.assertEqual(tiers["storage"]["bytes"], 0)
        self.assertEqual(tiers["device"]["bytes"], pin_budget // 2)
        self.assertTrue(projection["fits_resident"])

    def test_the_overflow_goes_to_host_before_storage(self):
        """Storage is the last resort by construction; a projection that skipped the host tier
        would understate a machine with lots of RAM and a slow disk, which is the common one."""
        profile = big_vram_machine()
        placement = per_token_projection(profile, 1)["placement"]
        size = placement["pin_budget_bytes"] + placement["host_cache_bytes"] + 4 * GB
        tiers = {tier["tier"]: tier for tier in per_token_projection(profile, size)["tiers"]}
        self.assertEqual(tiers["device"]["bytes"], placement["pin_budget_bytes"])
        self.assertEqual(tiers["host"]["bytes"], placement["host_cache_bytes"])
        self.assertEqual(tiers["storage"]["bytes"], 4 * GB)

    def test_the_tiers_account_for_every_byte_of_the_model(self):
        for machine, name in ((big_vram_machine(), "big"), (no_accelerator_machine(), "cpu")):
            for size in (1 * GB, 40 * GB, 700 * GB):
                with self.subTest(machine=name, size=size):
                    projection = per_token_projection(machine, size)
                    self.assertEqual(sum(t["bytes"] for t in projection["tiers"]), size,
                                     "a byte that belongs to no tier is a byte nobody paid for")

    def test_a_machine_with_no_device_pool_streams_everything(self):
        """The CPU backend has no device tier, and the projection must say so rather than
        pretending a fraction of the model is resident."""
        tiers = {t["tier"]: t for t in per_token_projection(no_accelerator_machine(), 8 * GB)["tiers"]}
        self.assertEqual(tiers["device"]["bytes"], 0)


class TestTheArithmetic(unittest.TestCase):
    def test_each_tier_costs_its_bytes_over_its_measured_bandwidth(self):
        profile = big_vram_machine(storage=storage(best_bytes_per_s=1.0e9))
        projection = per_token_projection(profile, 200 * GB)
        for tier in projection["tiers"]:
            with self.subTest(tier=tier["tier"]):
                self.assertAlmostEqual(tier["seconds"], tier["bytes"] / tier["bandwidth"], places=9)
        self.assertAlmostEqual(projection["seconds_per_token"],
                               sum(t["seconds"] for t in projection["tiers"]), places=9)

    def test_tokens_per_second_is_the_reciprocal_of_the_total(self):
        projection = per_token_projection(big_vram_machine(), 200 * GB)
        self.assertAlmostEqual(projection["tokens_per_second"],
                               1.0 / projection["seconds_per_token"], places=9)

    def test_the_storage_tier_is_charged_at_the_slower_of_read_and_link(self):
        """A byte read from disk still has to cross the link. Charging the read alone would report
        a machine with a fast NVMe behind a slow link as faster than it can possibly be."""
        profile = big_vram_machine(storage=storage(best_bytes_per_s=50e9),
                                   host_to_device_pinned_bandwidth=4e9,
                                   host_to_device_pageable_bandwidth=2e9)
        tiers = {t["tier"]: t for t in per_token_projection(profile, 400 * GB)["tiers"]}
        self.assertEqual(tiers["storage"]["bandwidth"], 4e9)

    def test_a_slow_disk_behind_a_fast_link_is_charged_at_the_disk(self):
        profile = big_vram_machine(storage=storage(best_bytes_per_s=5e8),
                                   host_to_device_pinned_bandwidth=25e9)
        tiers = {t["tier"]: t for t in per_token_projection(profile, 400 * GB)["tiers"]}
        self.assertEqual(tiers["storage"]["bandwidth"], 5e8)


class TestNothingIsInvented(unittest.TestCase):
    """The property that matters most: an unmeasured tier stays unmeasured."""

    def test_an_unprobed_disk_leaves_the_storage_tier_without_a_time(self):
        profile = big_vram_machine(storage=storage(best_bytes_per_s=None, error="not probed"))
        projection = per_token_projection(profile, 400 * GB)
        tiers = {t["tier"]: t for t in projection["tiers"]}
        self.assertGreater(tiers["storage"]["bytes"], 0)
        self.assertIsNone(tiers["storage"]["bandwidth"],
                          "the link speed must not stand in for a disk nobody timed")
        self.assertIsNone(tiers["storage"]["seconds"])
        self.assertFalse(projection["complete"])

    def test_an_incomplete_total_is_labelled_as_a_lower_bound(self):
        profile = big_vram_machine(storage=storage(best_bytes_per_s=None, error="not probed"))
        rendered = report(_collected(profile, 400 * GB))
        self.assertIn("INCOMPLETE", rendered)
        self.assertIn("at least", rendered)

    def test_a_machine_with_no_measured_link_still_reports_its_bytes(self):
        profile = big_vram_machine(host_to_device_pinned_bandwidth=None,
                                   host_to_device_pageable_bandwidth=None)
        tiers = {t["tier"]: t for t in per_token_projection(profile, 400 * GB)["tiers"]}
        self.assertGreater(tiers["host"]["bytes"], 0)
        self.assertIsNone(tiers["host"]["seconds"])

    def test_a_complete_projection_says_so(self):
        self.assertTrue(per_token_projection(big_vram_machine(), 400 * GB)["complete"])


class TestTheStorageVerdict(unittest.TestCase):
    def test_a_rotational_device_is_called_out_loudly(self):
        verdict = storage_health(big_vram_machine(storage=storage(rotational=True)))
        self.assertTrue(verdict["alarms"])
        self.assertIn("ROTATIONAL", " ".join(verdict["alarms"]),
                      "the one hardware fact most likely to explain a slow run has to shout")

    def test_slow_and_rotational_are_separate_alarms(self):
        """A fast disk behind a saturated link measures slow and is not rotational. Conflating the
        two would send someone to replace hardware that is fine."""
        slow = storage_health(big_vram_machine(
            storage=storage(rotational=False, best_bytes_per_s=40e6)))
        self.assertEqual(len(slow["alarms"]), 1)
        self.assertIn("MB/s", slow["alarms"][0])
        self.assertFalse(storage_health(big_vram_machine(storage=storage()))["alarms"])

    def test_a_slow_rotational_disk_raises_both(self):
        verdict = storage_health(big_vram_machine(
            storage=storage(rotational=True, best_bytes_per_s=40e6)))
        self.assertEqual(len(verdict["alarms"]), 2)

    def test_a_synthetic_probe_admits_it_did_not_read_the_real_shards(self):
        verdict = storage_health(big_vram_machine(
            storage=storage(probed_real_shards=False, synthetic_probe=True)))
        self.assertIn("temporary file", " ".join(verdict["alarms"]))

    def test_an_unprobed_path_says_what_to_pass(self):
        verdict = storage_health(big_vram_machine(
            storage=storage(error="no weights path given; storage was not probed")))
        self.assertIn("--weights-path", " ".join(verdict["alarms"]))


class TestTheCapabilityTable(unittest.TestCase):
    def setUp(self):
        caps.reset_announcements()
        caps.reset_caps_cache()

    tearDown = setUp

    def test_every_gate_reports_an_answer_and_a_fallback(self):
        for row in capability_rows(CPU):
            with self.subTest(capability=row["capability"]):
                self.assertIsInstance(row["answer"], bool, "a gate that will not decide is a bug")
                self.assertTrue(row["decided_by"].strip())
                self.assertTrue(row["fallback"].strip(),
                                "a capability with no defined fallback would crash on hardware "
                                "that lacks it")

    def test_a_capability_that_is_present_has_taken_no_fallback(self):
        for row in capability_rows(CPU):
            with self.subTest(capability=row["capability"]):
                self.assertEqual(row["taken"] is None, bool(row["answer"]))

    def test_the_bf16_row_names_the_silent_corruption_risk(self):
        """Falling back to fp16 is the one degradation whose symptom is wrong output rather than a
        slow run, so the table has to say that where someone will read it."""
        rows = {row["capability"]: row for row in
                capability_rows(CPU)}
        self.assertIn("SILENTLY", rows["bf16"]["fallback"])


class TestTheModelSize(unittest.TestCase):
    def test_a_checkpoint_on_disk_wins_over_everything_else(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "model.safetensors").write_bytes(b"x" * 4096)
            size, source = model_bytes_from(model=tmp, model_bytes=99, params=70e9)
            self.assertEqual(size, 4096)
            self.assertIn("measured", source)

    def test_only_weight_files_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "model.safetensors").write_bytes(b"x" * 2048)
            (Path(tmp) / "tokenizer.json").write_bytes(b"y" * 10_000)
            self.assertEqual(checkpoint_bytes(tmp), 2048)

    def test_a_directory_with_no_weights_measures_nothing_rather_than_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(checkpoint_bytes(tmp),
                              "zero would project a model that costs nothing to move")

    def test_a_parameter_count_is_scaled_by_the_declared_width(self):
        size, source = model_bytes_from(params=70_000_000_000, weight_bits=4)
        self.assertEqual(size, 35_000_000_000)
        self.assertIn("4 bits", source)

    def test_an_unstated_width_assumes_an_unquantized_checkpoint(self):
        self.assertEqual(model_bytes_from(params=1_000_000_000)[0], 2_000_000_000)

    def test_nothing_given_projects_nothing(self):
        self.assertEqual(model_bytes_from(), (None, None))


class TestThePackageInventory(unittest.TestCase):
    def test_every_package_reports_presence_as_a_boolean(self):
        for entry in doctor.package_inventory():
            with self.subTest(module=entry["module"]):
                self.assertIsInstance(entry["present"], bool)

    def test_the_inventory_covers_every_declared_optional_package(self):
        self.assertEqual([entry["module"] for entry in doctor.package_inventory()],
                         [package.module for package in OPTIONAL_PACKAGES])


class TestTheRenderedReport(unittest.TestCase):
    """It has to survive every machine, including the ones where almost nothing was measurable."""

    def machines(self):
        return {
            "big vram": big_vram_machine(),
            "no accelerator": no_accelerator_machine(),
            "nothing measured": make_profile(
                device_memory_bandwidth=None, host_to_device_pinned_bandwidth=None,
                host_to_device_pageable_bandwidth=None, device_total_bytes=None,
                device_free_bytes=None,
                storage=storage(best_bytes_per_s=None, queue_depth_1_bytes_per_s=None,
                                by_concurrency={}, saturating_concurrency=None,
                                rotational=None, error="storage was not probed")),
        }

    def test_it_renders_on_every_machine_with_and_without_a_model(self):
        for name, profile in self.machines().items():
            for size in (None, 40 * GB):
                with self.subTest(machine=name, model_bytes=size):
                    rendered = report(_collected(profile, size))
                    self.assertIn("CAPABILITY DECISIONS", rendered)
                    self.assertIn("OPTIONAL PACKAGES", rendered)
                    self.assertIn("WEIGHT STORAGE", rendered)
                    self.assertIn("PROJECTED COST PER TOKEN", rendered)

    def test_without_a_model_it_says_what_to_pass(self):
        rendered = report(_collected(big_vram_machine(), None))
        self.assertIn("--model PATH", rendered)

    def test_a_model_that_fits_is_reported_as_the_good_case(self):
        profile = big_vram_machine()
        rendered = report(_collected(profile, 1 * GB))
        self.assertIn("device-bandwidth bound", rendered)

    def test_the_json_form_carries_the_same_content(self):
        collected = _collected(big_vram_machine(), 40 * GB)
        payload = doctor.to_dict(collected)
        self.assertIsInstance(payload["profile"], dict)
        for key in ("capabilities", "packages", "storage", "quant_formats", "projection"):
            self.assertIn(key, payload)


class TestTheCommandLine(unittest.TestCase):
    def test_doctor_runs_end_to_end_on_the_cpu_backend(self):
        """The real thing, on the one backend every machine has."""
        from rocketllm.cli import main

        buffer = io.StringIO()
        collected = doctor.run(device="cpu", storage_budget_seconds=0.2, out=buffer)
        self.assertTrue(buffer.getvalue())
        self.assertEqual(collected["profile"].device_type, "cpu")
        self.assertEqual(main(["doctor", "--device", "cpu", "--storage-budget-seconds", "0.2"]), 0)

    def test_a_parameter_count_is_parsed_the_way_people_write_one(self):
        from rocketllm.cli import param_count

        self.assertEqual(param_count("70B"), 70_000_000_000)
        self.assertEqual(param_count("8b"), 8_000_000_000)
        self.assertEqual(param_count("1.5M"), 1_500_000)
        self.assertEqual(param_count("405000000"), 405_000_000)


def _collected(profile, model_bytes):
    """A collected diagnosis for a mocked machine, without probing the machine running the tests."""
    device_caps = CPU
    from rocketllm.quant import decision_table

    return {
        "profile": profile,
        "capabilities": capability_rows(device_caps),
        "packages": doctor.package_inventory(),
        "storage": storage_health(profile),
        "quant_formats": decision_table(caps=device_caps),
        "model_bytes": model_bytes,
        "model_bytes_source": "given as a byte count" if model_bytes else None,
        "projection": per_token_projection(profile, model_bytes) if model_bytes else None,
    }


if __name__ == "__main__":
    unittest.main(verbosity=2)
