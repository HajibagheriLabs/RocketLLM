"""The correctness gate, on the one backend every machine has.

tests/test_streaming_gpu.py --compare is the real gate, and it needs a GPU and a downloaded model,
which means it runs when someone remembers to run it. This is the same question asked in a form CI
can answer on a plain runner: build a tiny checkpoint, generate from it twice -- once through
transformers with the whole model loaded, once through RocketLLM streaming it a module at a time --
and require the two token sequences to be identical.

Identical, not close. Streaming changes *where* a weight is when the matmul reads it and nothing
else; a single differing token id means some path put a different number in front of the model, and
no tolerance would make that acceptable. A fast engine that produces wrong tokens is worth nothing.

Generation is covered here rather than a bare forward pass because it exercises what a forward pass
does not: the KV cache, the second and subsequent tokens replaying the same cyclic layer scan, and
residency surviving from one token to the next. Those are precisely the parts that a caching bug
would break while leaving a one-shot forward correct.

Everything runs on CPU and everything is built in a temporary directory. No accelerator, no network.
"""
import os
import tempfile
import unittest
from pathlib import Path

import torch

from rocketllm.base import RocketModel

#: CPU by default, because keeping that property is what lets this run in CI at all. Point it at an
#: accelerator to run the same comparison over the real transfer path.
DEVICE = os.environ.get("ROCKETLLM_TEST_DEVICE", "cpu")

PROMPT = torch.tensor([[1, 5, 9, 14, 3]])
NEW_TOKENS = 8


class StreamedModel(RocketModel):
    """RocketModel without the tokenizer, which these synthetic checkpoints have no reason to ship.

    The gate compares token ids, so a tokenizer would only be a download.
    """

    def get_tokenizer(self, hf_token=None):
        return None


def save(model, root):
    # fp32 throughout: this is a correctness comparison, and a reduced dtype would let a genuine
    # difference hide inside rounding that differs by summation order.
    model = model.to(torch.float32).eval()
    model.config.torch_dtype = "float32"
    model.save_pretrained(root, safe_serialization=True)
    return model


def dense_llama(root):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                         max_position_embeddings=64, tie_word_embeddings=False)
    return save(LlamaForCausalLM(config), root)


def mixtral_moe(root):
    from transformers import MixtralConfig, MixtralForCausalLM

    torch.manual_seed(0)
    config = MixtralConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                           num_local_experts=4, num_experts_per_tok=2,
                           max_position_embeddings=64, tie_word_embeddings=False)
    return save(MixtralForCausalLM(config), root)


class GenerationCase:
    """One checkpoint per class, and the reference sequence generated once from it.

    A plain mixin rather than a TestCase, so the shared cases are collected once per concrete
    architecture below instead of a third time against a class that has no checkpoint to build.
    """

    build = None

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "model"
        cls.root.mkdir(parents=True)
        reference = cls.build(cls.root).to(DEVICE)
        with torch.no_grad():
            cls.expected = reference(PROMPT.to(DEVICE)).logits
            # Greedy, so the reference is a function of the weights alone and nothing here depends
            # on a seed or on a sampler's implementation.
            cls.expected_sequence = reference.generate(
                PROMPT.to(DEVICE), max_new_tokens=NEW_TOKENS, do_sample=False).tolist()
        del reference

    @classmethod
    def tearDownClass(cls):
        tmp = getattr(cls, "_tmp", None)
        if tmp is not None:
            tmp.cleanup()

    def stream(self):
        return StreamedModel(str(self.root), device=DEVICE, dtype=torch.float32)

    def generate(self, model):
        with torch.no_grad():
            out = model.generate(PROMPT.to(DEVICE), max_new_tokens=NEW_TOKENS, do_sample=False)
        return (out if isinstance(out, torch.Tensor) else out.sequences).tolist()

    def assert_matches(self, produced, what):
        self.assertEqual(produced, self.expected_sequence,
                         f"{what} produced different tokens from a full load of the same weights. "
                         f"Streaming may change where a weight lives, never what the model computes")

    # -- the gate itself -------------------------------------------------------------------------

    def test_streamed_generation_matches_a_full_load(self):
        model = self.stream()
        try:
            self.assert_matches(self.generate(model), "the streamed run")
        finally:
            model.close()

    def test_the_logits_are_identical_before_a_single_token_is_sampled(self):
        """Localises a failure: wrong logits is an arithmetic bug, right logits and wrong tokens is
        a generation-loop or KV cache bug, and they are fixed in different files."""
        model = self.stream()
        try:
            with torch.no_grad():
                got = model(PROMPT.to(DEVICE)).logits
            self.assertTrue(torch.equal(got, self.expected),
                            f"streamed logits differ by up to "
                            f"{(got - self.expected).abs().max().item():.3e}")
        finally:
            model.close()

    def test_a_second_generation_from_the_same_model_matches_too(self):
        """The cache deliberately keeps its residency across generations. That must be invisible in
        the output: what is retained is where the bytes are, never a piece of the last run."""
        model = self.stream()
        try:
            self.assert_matches(self.generate(model), "the first generation")
            self.assert_matches(self.generate(model), "the second generation")
        finally:
            model.close()

    def test_generation_after_reset_matches(self):
        """reset() drops residency, so the next generation re-reads everything from storage. Same
        weights, same path, same answer -- or the storage tier is handing back something else."""
        model = self.stream()
        try:
            self.assert_matches(self.generate(model), "the generation before reset")
            model.reset()
            self.assert_matches(self.generate(model), "the generation after reset")
        finally:
            model.close()

    def test_it_runs_with_nothing_pinned_at_all(self):
        """Pure streaming is what a device with no spare memory gets, and it is the configuration
        most likely to be reached in the field. It has to produce the same tokens as any other."""
        model = StreamedModel(str(self.root), device=DEVICE, dtype=torch.float32,
                              pin_policy="off", host_cache_gb=0)
        try:
            self.assertTrue(model.cache.pinned == set() or not model.cache.pinned)
            self.assert_matches(self.generate(model), "the pure-streaming run")
        finally:
            model.close()


class TestDenseGeneration(GenerationCase, unittest.TestCase):
    build = staticmethod(dense_llama)


class TestMixtureGeneration(GenerationCase, unittest.TestCase):
    """A mixture streams experts rather than whole layers, which is a different code path end to
    end -- routing, per-expert residency, and a separate cache policy."""

    build = staticmethod(mixtral_moe)


if __name__ == "__main__":
    unittest.main(verbosity=2)
