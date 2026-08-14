"""Tests for running a checkpoint in a dtype other than the one it was saved in.

`dtype=` is documented as a supported override, and until this file existed it was only ever
exercised with a value that happened to equal the checkpoint's own. That is what hid the bug: the
model is built from the config, so every meta placeholder carries the config's dtype, and
accelerate's placement casts an incoming value to the existing parameter's dtype whenever it is not
told otherwise. When the two agree that cast is invisible. When they differ it silently undoes the
decision the transfer planner just made, and the run dies inside the first matmul with a message
naming two dtypes and nothing pointing back at the cause.

So the cases here are all mismatch cases, in both directions -- widening a half-precision checkpoint
and narrowing a full-precision one -- plus the property the second half of the fix protects: a
coalesced layer has to stay ONE allocation bound as views, which is the whole reason for packing it,
and the implicit cast quietly replaced every view with a copy.

CPU by default, like the rest of the streaming tests, so this keeps running on a plain CI runner.
"""
import gc
import os
import tempfile
import unittest
from pathlib import Path

import torch

from rocketllm.base import RocketModel

PROMPT = torch.tensor([[1, 5, 9, 14, 3]])
DEVICE = os.environ.get("ROCKETLLM_TEST_DEVICE", "cpu")


def _temporary_directory_ignoring_cleanup_errors():
    """A TemporaryDirectory that survives a file it cannot unlink, on every supported Python.

    ``ignore_cleanup_errors`` arrived in 3.10 and this project supports 3.9, where passing it is a
    TypeError. On 3.9 the flag is simply absent and cleanup failures surface as they always did --
    which is why the ignoring exists rather than being the default: see the caller.
    """
    try:
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    except TypeError:
        return tempfile.TemporaryDirectory()


class StreamedModel(RocketModel):
    """RocketModel without the tokenizer, which these checkpoints have no reason to ship."""

    def get_tokenizer(self, hf_token=None):
        return None


def build(root, saved_dtype):
    """A small dense model written to disk in `saved_dtype`."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(11)
    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                         max_position_embeddings=64, tie_word_embeddings=False)
    model = LlamaForCausalLM(config).eval().to(saved_dtype)
    model.config.torch_dtype = str(saved_dtype).replace("torch.", "")
    model.save_pretrained(root, safe_serialization=True)


def reference(root, run_dtype):
    """The same checkpoint loaded whole, in the dtype the streamed run will use.

    Loaded from disk rather than kept from `build`, and that is not fussiness. Casting a live model
    to half precision converts its BUFFERS too, so a rotary inv_freq that was computed in float32
    comes back from a round trip through fp16 with fp16's precision. The streaming path never does
    that -- it builds the model at the runtime dtype and computes inv_freq fresh -- so a reference
    built by casting would differ by a ULP in the logits for a reason that has nothing to do with
    what is being tested.
    """
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(root, dtype=run_dtype)
    except TypeError:      # transformers renamed torch_dtype -> dtype in 4.56
        model = AutoModelForCausalLM.from_pretrained(root, torch_dtype=run_dtype)
    return model.eval().to(DEVICE)


def during_forward(model, layer_index=0):
    """What a streamed layer's parameters look like WHILE it runs.

    Inspecting them afterwards says nothing: on a device with no spare budget the cache holds
    nothing, so the weights are unbound the moment the layer finishes and every parameter reads back
    as meta. A test that looked then would pass by finding nothing to check. This rides a pre-hook
    registered after the engine's own, so it sees the layer exactly as the matmul does.
    """
    seen = {}
    layer = model.model.model.layers[layer_index]

    def capture(module, args):
        seen.update({name: (p.dtype, p.untyped_storage().data_ptr())
                     for name, p in module.named_parameters()
                     if p.device.type != "meta"})

    handle = layer.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            model(PROMPT.to(DEVICE))
    finally:
        handle.remove()
    assert seen, "the capture hook saw no bound parameters at all"
    return seen


class DtypeCase(unittest.TestCase):
    """One checkpoint per class, since saving and splitting dominates the runtime."""

    saved_dtype = torch.float16
    run_dtype = torch.float32

    @classmethod
    def setUpClass(cls):
        # safetensors memory-maps what it reads, and Windows will not unlink a mapped file, so the
        # reference model holding one can outlive the attempt to remove the directory. Ignoring that
        # keeps a temp-directory detail from failing a test that has already passed.
        cls._tmp = _temporary_directory_ignoring_cleanup_errors()
        cls.root = Path(cls._tmp.name) / "model"
        cls.root.mkdir(parents=True)
        build(cls.root, cls.saved_dtype)
        cls.reference = reference(cls.root, cls.run_dtype)

    @classmethod
    def tearDownClass(cls):
        cls.reference = None
        gc.collect()
        tmp = getattr(cls, "_tmp", None)
        if tmp is not None:
            try:
                tmp.cleanup()
            except OSError:
                # Python 3.9 has no ignore_cleanup_errors, so the same mapped-file case it exists
                # for is caught here instead. A temp directory the OS is still holding open is not
                # a result: the assertions have already run.
                pass

    def stream(self, dtype=None):
        """`dtype=False` means "pass nothing", which is different from passing the same value."""
        if dtype is False:
            return StreamedModel(str(self.root), device=DEVICE)
        return StreamedModel(str(self.root), device=DEVICE,
                             dtype=self.run_dtype if dtype is None else dtype)


class TestWideningAHalfPrecisionCheckpoint(DtypeCase):
    """float16 on disk, float32 at runtime. The reported crash, and the case worth having:
    float32 is what makes greedy speculative decoding bit-exact against plain decoding."""

    saved_dtype = torch.float16
    run_dtype = torch.float32

    def test_a_forward_runs_at_all(self):
        """The regression. This raised `expected mat1 and mat2 to have the same dtype`."""
        model = self.stream()
        try:
            with torch.no_grad():
                logits = model(PROMPT.to(DEVICE)).logits
            self.assertEqual(logits.dtype, torch.float32)
        finally:
            model.close()

    def test_every_streamed_parameter_lands_in_the_runtime_dtype(self):
        model = self.stream()
        try:
            seen = during_forward(model)
            wrong = [name for name, (dtype, _) in seen.items() if dtype != self.run_dtype]
            self.assertEqual(wrong, [], "these parameters kept the checkpoint's dtype")
            self.assertGreaterEqual(len(seen), 7, "too few parameters seen to mean anything")
        finally:
            model.close()

    def test_the_result_matches_the_same_weights_run_without_streaming(self):
        """Streaming must not change the arithmetic, only where the weights live -- and that has to
        hold when it is also changing their dtype on the way."""
        model = self.stream()
        try:
            with torch.no_grad():
                got = model(PROMPT.to(DEVICE)).logits
                expected = self.reference(PROMPT.to(DEVICE)).logits
            self.assertTrue(torch.equal(got, expected),
                            f"streamed logits differ by up to "
                            f"{(got - expected).abs().max().item():.3e}")
        finally:
            model.close()

    def test_the_model_reports_the_dtype_it_is_actually_running_in(self):
        """The root cause, stated directly. The model is built from the config, so a config saying
        one thing while the engine intends another is what every downstream default reads."""
        model = self.stream()
        try:
            self.assertEqual(model.config.torch_dtype, self.run_dtype)
            for _, (dtype, _) in during_forward(model).items():
                self.assertEqual(dtype, self.run_dtype)
        finally:
            model.close()

    def test_a_coalesced_layer_is_still_one_allocation(self):
        """The second half of the fix, and the reason it is not merely belt-and-braces.

        A layer's tensors are packed into one staging buffer and bound as views into it -- one
        allocation per layer, no second copy. Left to default, accelerate's cast-to-the-existing-
        dtype replaced every one of those views with a fresh tensor, so a dtype override silently
        turned the coalesced path back into the per-tensor path it exists to replace. Views share a
        storage; copies do not.
        """
        model = self.stream()
        try:
            seen = during_forward(model)
            pointers = {pointer for name, (_, pointer) in seen.items() if "norm" not in name}
            self.assertEqual(len(pointers), 1,
                             f"a coalesced layer bound {len(pointers)} separate allocations; the "
                             f"parameters are copies rather than views into the staged buffer")
        finally:
            model.close()


class TestNarrowingAFullPrecisionCheckpoint(DtypeCase):
    """float32 on disk, float16 at runtime. The other direction, which has to work too -- it is
    what someone with a fp32 checkpoint and a small card asks for."""

    saved_dtype = torch.float32
    run_dtype = torch.float16

    def test_a_forward_runs_and_the_weights_are_narrowed(self):
        model = self.stream()
        try:
            seen = during_forward(model)
            self.assertTrue(all(dtype == torch.float16 for dtype, _ in seen.values()),
                            f"some parameters kept float32: {seen}")
        finally:
            model.close()

    def test_the_result_still_matches_the_same_weights_run_without_streaming(self):
        model = self.stream()
        try:
            with torch.no_grad():
                got = model(PROMPT.to(DEVICE)).logits
                expected = self.reference(PROMPT.to(DEVICE)).logits
            self.assertEqual(got.dtype, torch.float16)
            self.assertTrue(torch.equal(got, expected),
                            f"streamed logits differ by up to "
                            f"{(got - expected).abs().max().item():.3e}")
        finally:
            model.close()


class TestTheDefaultIsUnchanged(DtypeCase):
    """No override at all: the path every other test in the suite exercises, kept honest here.

    The fix touches how the runtime dtype reaches the model, so the case where nothing was
    overridden has to be shown to still take the checkpoint's own dtype rather than quietly
    acquiring a new one.
    """

    saved_dtype = torch.float32
    run_dtype = torch.float32

    def test_no_dtype_argument_keeps_the_checkpoints_dtype(self):
        model = self.stream(dtype=False)
        try:
            self.assertEqual(model.running_dtype, torch.float32)
            seen = during_forward(model)
            self.assertTrue(all(dtype == torch.float32 for dtype, _ in seen.values()))
            with torch.no_grad():
                got = model(PROMPT.to(DEVICE)).logits
                expected = self.reference(PROMPT.to(DEVICE)).logits
            self.assertTrue(torch.equal(got, expected))
        finally:
            model.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
