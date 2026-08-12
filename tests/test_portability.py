"""The portability matrix: RocketLLM on hardware nobody running this test owns.

RocketLLM has no reference machine and its contributors have wildly different ones, so the property
that actually has to hold is not "it works here" but "it works everywhere, and where it cannot do
something fast it does it slowly instead of falling over". That property is untestable on one box --
unless the box is made to behave like the others, which is what this file does.

Two dimensions are swept, because they fail independently.

**How much of the model fits.** Four devices: one that holds the whole thing, one that holds about
half, one with room for exactly a single layer, and one whose pin budget works out to zero. Each is
emulated by a HardwareProfile whose measurements are supplied rather than probed -- the derived
knobs are what the cache sizes itself from, so a supplied profile *is* a different machine as far as
the engine can tell. Where a real CUDA device is present, `cap_vram` caps the process on top of
that, so the same case additionally runs against an allocator that will genuinely refuse.

**Which capabilities the device has.** Pinned host memory, async copy streams, fused 4-bit kernels
and bf16, each removed on its own. Every one of them has a defined fallback in docs/HARDWARE.md,
and every fallback has to produce the same tokens as the capability would have.

The assertion in every case is the same and it is not negotiable: the generated token ids are
identical to a full load of the same weights. Around it sit assertions about *how* the cache
degraded -- window size, what got pinned, how much had to be re-read -- because a run that produced
the right answer by quietly streaming everything on the machine that should have kept it resident is
a regression that correctness alone would not catch.

Nothing here needs an accelerator.
"""
import contextlib
import dataclasses
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from rocketllm import base as base_module
from rocketllm.base import RocketModel
from rocketllm.hw import caps as C
from rocketllm.hw.caps import CpuCaps, FusedPlan
from rocketllm.hw.profile import HardwareProfile

# cap_vram lives with the manual GPU harness next door, and the mocked machines with the profile
# tests. This file is run both through pytest and directly from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_hw_profile import make_profile  # noqa: E402
from test_streaming_gpu import cap_vram  # noqa: E402

DEVICE = os.environ.get("ROCKETLLM_TEST_DEVICE", "cpu")

PROMPT = torch.tensor([[1, 5, 9, 14, 3]])
NEW_TOKENS = 6
GB = 1024 ** 3


# ---------------------------------------------------------------------------------------------
# the checkpoint
# ---------------------------------------------------------------------------------------------

class StreamedModel(RocketModel):
    """RocketModel without the tokenizer; these synthetic checkpoints have no reason to ship one."""

    def get_tokenizer(self, hf_token=None):
        return None


def build_checkpoint(root, dtype=torch.float32):
    """A small dense model, deep enough that residency and eviction are distinguishable.

    Several layers rather than two on purpose: with one or two, "holds half the model" and "holds
    one layer" are the same device, and the middle of the matrix would test nothing.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                         max_position_embeddings=64, tie_word_embeddings=False)
    model = LlamaForCausalLM(config).to(dtype).eval()
    model.config.torch_dtype = str(dtype).replace("torch.", "")
    model.save_pretrained(root, safe_serialization=True)
    return model


def reference_sequence(model):
    with torch.no_grad():
        return model.generate(PROMPT.to(DEVICE), max_new_tokens=NEW_TOKENS,
                              do_sample=False).tolist()


# ---------------------------------------------------------------------------------------------
# emulating a machine
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class EmulatedDevice:
    """A machine described by what the engine would have measured on it.

    `usable_bytes` and `window_fraction` are the two numbers everything downstream comes from: the
    prefetch window takes its share of usable device memory first, and what is left is the pin
    budget. Expressing the case this way rather than as "a 4GB card" keeps it about the quantity
    that actually decides the engine's behaviour -- how the model's size compares to the budget.
    """

    name: str
    usable_bytes: int
    window_fraction: float
    host_cache_bytes: int = 0

    @property
    def window_budget_bytes(self):
        return int(self.usable_bytes * self.window_fraction)

    @property
    def pin_budget_bytes(self):
        return max(0, self.usable_bytes - self.window_budget_bytes)

    def profile(self):
        # reserve is pinned at zero so that usable device memory is exactly what this case asked
        # for: the derivation is usable = total - reserve, and a measured reserve would make the
        # case's own number approximate.
        profile = make_profile(device_total_bytes=self.usable_bytes,
                               device_free_bytes=self.usable_bytes)
        profile.derive(overrides={
            "reserve_bytes": 0,
            "window_fraction": self.window_fraction,
            "host_cache_bytes": self.host_cache_bytes,
            "staging_pool_bytes": 0,
            "io_workers": 1,
        })
        return profile

    @property
    def vram_cap_gb(self):
        """What to cap a real CUDA process at to match this case, where one is present."""
        return max(0.25, self.usable_bytes / GB)


class EmulatedCaps(CpuCaps):
    """A device whose capability answers are dictated rather than queried.

    Subclassing the CPU backend rather than inventing one keeps every fallback surface real: the
    streams, events and staging buffers handed out here are the ones a machine without those
    features genuinely gets, so a test that passes proves the fallback works and not merely that a
    stub was accepted.
    """

    def __init__(self, device, pinned=False, streams=False, fused=False, bf16=True, fp16=True):
        super().__init__(device)
        self._pinned = pinned
        self._streams = streams
        self._fused = fused
        self._bf16 = bf16
        self._fp16 = fp16

    @property
    def can_pin_memory(self):
        return self._pinned

    @property
    def has_async_streams(self):
        return self._streams

    @property
    def supports_bf16(self):
        return self._bf16

    @property
    def supports_fp16(self):
        return self._fp16

    def fused_4bit_plan(self):
        if self._fused:
            return FusedPlan("fused_packed", "emulated_kernel",
                             "emulated: a fused 4-bit kernel is available on this device")
        return FusedPlan("dequant_to_scratch", None,
                         "emulated: no fused 4-bit kernel, so packed weights are expanded into a "
                         "reusable scratch buffer and computed in the compute dtype")


@contextlib.contextmanager
def emulate(device=None, caps=None):
    """Run the block as though the machine were the one described.

    Both seams are patched on :mod:`rocketllm.base` rather than deeper, because that is where the
    engine reaches for them -- the profile it sizes the cache from, and the capability object it
    asks what this hardware can do. Everything below those two calls is the real code path.
    """
    profile = device.profile() if device is not None else None

    class _Profile:
        @staticmethod
        def load_or_probe(*args, **kwargs):
            if profile is None:
                return HardwareProfile.load_or_probe(*args, **kwargs)
            return profile

    def _get_caps(dev, announce=True):
        if caps is None:
            return C.get_caps(dev, announce=announce)
        built = caps(torch.device(dev) if not isinstance(dev, torch.device) else dev)
        if announce:
            # The real get_caps announces when a device is first resolved, and several of the
            # degradations below are checked by the announcement they make.
            built.announce_degradations()
        return built

    saved_profile = base_module.HardwareProfile
    saved_caps = base_module.get_caps
    base_module.HardwareProfile = _Profile
    base_module.get_caps = _get_caps

    # A real CUDA device is capped as well as described, so on a developer's machine the same case
    # runs against an allocator that will actually refuse. On every other backend this is a no-op
    # that says so, which is why it is safe to call unconditionally.
    capped = device is not None and torch.cuda.is_available() and DEVICE.startswith("cuda")
    if capped:
        cap_vram(device.vram_cap_gb)
    try:
        yield profile
    finally:
        base_module.HardwareProfile = saved_profile
        base_module.get_caps = saved_caps
        if capped:
            torch.cuda.set_per_process_memory_fraction(1.0, 0)


# ---------------------------------------------------------------------------------------------
# the matrix
# ---------------------------------------------------------------------------------------------

class PortabilityCase(unittest.TestCase):
    """One checkpoint, its full-load reference, and the plumbing to run it on a described machine."""

    dtype = torch.float32

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "model"
        cls.root.mkdir(parents=True)
        reference = build_checkpoint(cls.root, dtype=cls.dtype).to(DEVICE)
        cls.expected = reference_sequence(reference)
        del reference
        # Measured once, from a load with a budget large enough to hold everything, so the cases
        # below can be expressed against the model's real size rather than a guess at it.
        cls.model_bytes, cls.largest_layer_bytes = cls._measure()

    @classmethod
    def _measure(cls):
        roomy = EmulatedDevice("measuring", usable_bytes=1 * GB, window_fraction=0.5)
        with emulate(roomy):
            model = StreamedModel(str(cls.root), device=DEVICE, dtype=cls.dtype)
            try:
                layers = [model._layer_packed_bytes(i) for i in model._streamed_indices]
                total = sum(layers) + sum(model._unit_byte_counts.values())
                return total, max(layers)
            finally:
                model.close()

    @classmethod
    def tearDownClass(cls):
        tmp = getattr(cls, "_tmp", None)
        if tmp is not None:
            tmp.cleanup()

    def setUp(self):
        C.reset_announcements()
        C.reset_caps_cache()

    tearDown = setUp

    @contextlib.contextmanager
    def streamed(self, device=None, caps=None, **kwargs):
        # dtype is settable rather than fixed because passing one explicitly is an instruction the
        # engine obeys: the dtype degradation only happens for a run that did NOT name a dtype and
        # therefore took the checkpoint's, which is the case a user is actually in.
        kwargs.setdefault("dtype", self.dtype)
        with emulate(device, caps):
            model = StreamedModel(str(self.root), device=DEVICE, **kwargs)
            try:
                yield model
            finally:
                model.close()

    def generate(self, model):
        with torch.no_grad():
            out = model.generate(PROMPT.to(DEVICE), max_new_tokens=NEW_TOKENS, do_sample=False)
        return (out if isinstance(out, torch.Tensor) else out.sequences).tolist()

    def assert_correct(self, model, case):
        self.assertEqual(self.generate(model), self.expected,
                         f"{case} produced different tokens from a full load of the same weights. "
                         f"A device's size and capabilities decide where weights live and how fast "
                         f"they get there -- never what the model computes")


class TestDeviceSizes(PortabilityCase):
    """The four devices the matrix is really about, from roomy to nothing to spare."""

    def devices(self):
        model, layer = self.model_bytes, self.largest_layer_bytes
        return {
            # Four times the model, a quarter of it committed to the window: everything else can be
            # pinned, so nothing should ever be re-read.
            "whole model fits": EmulatedDevice(
                "whole model fits", usable_bytes=model * 4, window_fraction=0.25,
                host_cache_bytes=model),
            # Usable memory equal to the model with half committed to the window, so about half the
            # weights can stay resident and the rest cannot.
            "about half fits": EmulatedDevice(
                "about half fits", usable_bytes=model, window_fraction=0.5,
                host_cache_bytes=model),
            # Room for exactly one layer, all of it window. This is the floor the engine is allowed
            # to reach: a cache that cannot hold a single layer cannot run a forward at all.
            "one layer fits": EmulatedDevice(
                "one layer fits", usable_bytes=layer, window_fraction=1.0,
                host_cache_bytes=model),
            # The window takes everything, so the pin budget arithmetic yields zero. This is a
            # supported configuration, not an error: it is what a card with no spare memory gets.
            "pin budget is zero": EmulatedDevice(
                "pin budget is zero", usable_bytes=layer * 3, window_fraction=1.0,
                host_cache_bytes=model),
            # The same, with no host tier either: every eviction drops straight to storage and every
            # token re-reads it. The slowest configuration this engine has, and a supported one.
            "pure streaming": EmulatedDevice(
                "pure streaming", usable_bytes=layer * 2, window_fraction=1.0,
                host_cache_bytes=0),
        }

    def test_every_device_generates_the_same_tokens(self):
        for name, device in self.devices().items():
            with self.subTest(device=name):
                with self.streamed(device) as model:
                    self.assert_correct(model, name)

    def test_the_cache_is_sized_to_the_device_it_was_given(self):
        """A case that silently got a different budget would prove nothing about the device it
        claims to describe, and every other assertion here rests on this one."""
        for name, device in self.devices().items():
            with self.subTest(device=name):
                with self.streamed(device) as model:
                    report = model.cache.report()
                    self.assertEqual(report["device_capacity"], device.usable_bytes)
                    self.assertEqual(report["host_capacity"], device.host_cache_bytes)

    def test_a_roomy_device_pins_and_a_tight_one_does_not(self):
        devices = self.devices()
        with self.streamed(devices["whole model fits"]) as model:
            self.assertEqual(len(model.cache.pinned), len(model._streamed_indices),
                             "a device with room for the whole model must keep all of it resident")
        with self.streamed(devices["about half fits"]) as model:
            pinned = len(model.cache.pinned)
            self.assertGreater(pinned, 0, "half a model's worth of budget must hold half a model")
            self.assertLess(pinned, len(model._streamed_indices))

    def test_a_zero_pin_budget_pins_nothing_and_still_runs(self):
        for name in ("one layer fits", "pin budget is zero", "pure streaming"):
            with self.subTest(device=name):
                with self.streamed(self.devices()[name]) as model:
                    self.assertEqual(model._pin_budget, 0)
                    self.assertEqual(len(model.cache.pinned), 0)
                    self.assert_correct(model, name)

    def test_a_device_with_room_for_one_layer_holds_a_window_of_one(self):
        with self.streamed(self.devices()["one layer fits"]) as model:
            self.assertEqual(model.cache.window, 1,
                             "the window has to shrink to one rather than refuse to run: one layer "
                             "resident is the minimum a forward pass can be done with")
            self.assert_correct(model, "one layer fits")

    def test_less_device_memory_means_more_re_reading_and_never_less(self):
        """The whole point of the cache, stated as an inequality.

        A tighter device is allowed to be slower. It is not allowed to be *faster*, and a run that
        re-read no more on a one-layer device than on one holding the whole model would mean the
        residency was doing nothing on either.
        """
        fetches = {}
        for name, device in self.devices().items():
            with self.streamed(device) as model:
                self.generate(model)
                fetches[name] = model.cache.report()["fetches"]

        order = ["whole model fits", "about half fits", "one layer fits", "pure streaming"]
        for tighter, roomier in zip(order[1:], order):
            with self.subTest(tighter=tighter, roomier=roomier):
                self.assertGreaterEqual(fetches[tighter], fetches[roomier],
                                        f"{tighter} re-read less than {roomier}: {fetches}")
        self.assertGreater(fetches["pure streaming"], fetches["whole model fits"],
                           f"residency bought nothing at all: {fetches}")

    def test_a_resident_model_is_read_once_and_never_again(self):
        """The best case has to actually be the best case: nothing re-read after the first pass."""
        with self.streamed(self.devices()["whole model fits"]) as model:
            self.generate(model)
            after_first = model.cache.report()["fetches"]
            self.generate(model)
            self.assertEqual(model.cache.report()["fetches"], after_first,
                             "a model that fits entirely on the device was re-read anyway")


class TestMissingCapabilities(PortabilityCase):
    """Each capability removed on its own, against the fallback docs/HARDWARE.md promises for it."""

    #: Roomy enough that nothing here is confounded by the device also being short of memory.
    def roomy(self):
        return EmulatedDevice("capability sweep", usable_bytes=self.model_bytes * 4,
                              window_fraction=0.25, host_cache_bytes=self.model_bytes)

    def caps_factory(self, **flags):
        settings = dict(pinned=True, streams=True, fused=True, bf16=True, fp16=True)
        settings.update(flags)
        return lambda device: EmulatedCaps(device, **settings)

    def test_no_pinned_memory_uses_pageable_buffers_and_the_same_tokens(self):
        with self.streamed(self.roomy(), self.caps_factory(pinned=False)) as model:
            self.assertFalse(model.caps.can_pin_memory)
            buffer = model.caps.pinned_empty((16,), torch.uint8)
            self.assertFalse(buffer.is_pinned(),
                             "a device without pinned memory must get an ordinary host buffer, "
                             "not an error")
            self.assert_correct(model, "a device with no pinned host memory")

    def test_no_async_streams_uses_the_synchronous_transfer_path(self):
        with self.streamed(self.roomy(), self.caps_factory(streams=False)) as model:
            stream = model.caps.copy_stream()
            self.assertFalse(stream.is_async,
                             "without copy streams the transfer path has to be the synchronous one")
            # The synchronous stand-in still has to present the whole surface, or the streaming
            # path would need a branch of its own for this backend.
            with stream:
                stream.wait_event(stream.record_event())
            stream.synchronize()
            self.assert_correct(model, "a device with no async copy streams")

    def test_no_fused_kernels_dequantizes_into_scratch(self):
        with self.streamed(self.roomy(), self.caps_factory(fused=False)) as model:
            plan = model.caps.fused_4bit_plan()
            self.assertEqual(plan.path, "dequant_to_scratch")
            self.assertFalse(plan.fused)
            self.assert_correct(model, "a device with no fused 4-bit kernels")

    def test_a_fused_kernel_being_present_changes_nothing_about_the_answer(self):
        """The other half of the same claim: the fused path is a speed decision, not a numeric one."""
        with self.streamed(self.roomy(), self.caps_factory(fused=True)) as model:
            self.assertTrue(model.caps.fused_4bit_plan().fused)
            self.assert_correct(model, "a device with fused 4-bit kernels")

    def test_every_missing_capability_is_announced_exactly_once(self):
        """Announced once per process, never per layer: a streaming run touches every module
        hundreds of times per token, and a per-layer notice would bury the console."""
        factory = self.caps_factory(pinned=False, streams=False, fused=False)
        with self.assertLogs("rocketllm.hw.caps", level=logging.INFO) as captured:
            with self.streamed(self.roomy(), factory) as model:
                self.generate(model)
        pinned = [line for line in captured.output if "pinned host memory is unavailable" in line]
        streams = [line for line in captured.output if "async copy streams are unavailable" in line]
        self.assertEqual(len(pinned), 1, captured.output)
        self.assertEqual(len(streams), 1, captured.output)


class TestMissingBf16(PortabilityCase):
    """bf16 is the one absence whose symptom is wrong output rather than a slow run.

    The checkpoint declares bf16, so the engine asks for it and cannot have it. What has to happen
    is a fallback to fp16, a warning that names the risk in words a user will recognise when their
    output looks wrong, and -- since fp16 is what the run is now in -- tokens matching a full load
    at fp16. The comparison is against fp16 rather than against the bf16 reference on purpose:
    streaming must not change the arithmetic, but a narrower dtype legitimately does, and pretending
    otherwise would be the kind of test that passes by asserting the wrong thing.
    """

    dtype = torch.bfloat16

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from transformers import AutoModelForCausalLM

        # A second reference at the dtype the fallback lands in, loaded from the same files the
        # streamed run reads rather than rebuilt, so the two cannot differ by a weight.
        reference = AutoModelForCausalLM.from_pretrained(cls.root).to(torch.float16).to(DEVICE)
        cls.expected_fp16 = reference_sequence(reference)
        del reference

    def roomy(self):
        return EmulatedDevice("no bf16", usable_bytes=self.model_bytes * 4, window_fraction=0.25,
                              host_cache_bytes=self.model_bytes)

    def without_bf16(self, fp16=True):
        def factory(device):
            return EmulatedCaps(device, pinned=False, streams=False, fused=False, bf16=False,
                                fp16=fp16)
        return factory

    def test_the_run_falls_back_to_fp16_and_says_why(self):
        # dtype=None so the engine takes the checkpoint's declared bf16 and has to degrade it. A
        # dtype named on the call is an instruction, and would be obeyed rather than degraded.
        with self.assertLogs("rocketllm.hw.caps", level=logging.WARNING) as captured:
            with self.streamed(self.roomy(), self.without_bf16(), dtype=None) as model:
                self.assertIs(model.running_dtype, torch.float16)
                sequence = self.generate(model)
        warning = " ".join(captured.output)
        self.assertIn("bf16 is not supported", warning)
        self.assertIn("corrupts output silently", warning,
                      "the warning has to name the symptom, because the symptom is plausible "
                      "wrong tokens rather than an error anyone would see")
        self.assertEqual(sequence, self.expected_fp16,
                         "the fp16 fallback must still stream the same weights it would have")

    def test_a_device_with_neither_bf16_nor_fp16_runs_in_fp32(self):
        """The bottom of the dtype ladder. Correct, twice the bytes per token, and it must run."""
        with self.streamed(self.roomy(), self.without_bf16(fp16=False), dtype=None) as model:
            self.assertIs(model.running_dtype, torch.float32)
            self.generate(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
