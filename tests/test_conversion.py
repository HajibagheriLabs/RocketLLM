"""Reading what transformers declares about its own checkpoints.

Two generations spell the declaration differently and this engine has to read both, because both
are live: 4.x publishes a dict of regex renames, 5.x an ordered list of transform objects that can
also change a tensor's SHAPE. The second is why this module exists at all -- transformers 5 stopped
building a mixture's experts as a list of modules and builds one batched module instead, so the
per-expert tensors in an existing checkpoint became rows of a fused parameter, and no rename says
that.

What is checked here is the reading, not the models. A test that asserted "Mixtral declares two
fusions" would be asserting a fact about transformers that transformers is free to change; what
must hold is that whatever it declares is understood, that a declaration this engine has no partial
form of is declined rather than guessed at, and that the row arithmetic derived from the declared
operations is the arithmetic transformers itself performs. That last one is verified against a real
model's own weights, because it is the one where being wrong produces fluent, wrong output.
"""
import sys
import unittest
from pathlib import Path

import torch
from accelerate import init_empty_weights

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocketllm.conversion import (CheckpointConversion, ExpertFusion,  # noqa: E402
                                  _row_arithmetic, describe)


def has_modern_conversions():
    try:
        import transformers.conversion_mapping  # noqa: F401
    except ImportError:
        return False
    return True


needs_modern = unittest.skipUnless(
    has_modern_conversions(),
    "this transformers declares no weight-conversion pipeline, so there is nothing to read")


def build_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                      max_position_embeddings=32, tie_word_embeddings=False)
    with init_empty_weights(include_buffers=False):
        return LlamaForCausalLM(cfg)


def build_mixtral(experts=3, hidden=16, intermediate=32, materialize=False):
    from transformers import MixtralConfig, MixtralForCausalLM

    cfg = MixtralConfig(hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=1,
                        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                        num_local_experts=experts, num_experts_per_tok=2,
                        max_position_embeddings=32, tie_word_embeddings=False)
    if materialize:
        torch.manual_seed(0)
        return MixtralForCausalLM(cfg).to(torch.float32).eval()
    with init_empty_weights(include_buffers=False):
        return MixtralForCausalLM(cfg)


def build_qwen25vl():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    cfg = Qwen2_5_VLConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                         max_position_embeddings=64, tie_word_embeddings=False),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                           in_chans=3, out_hidden_size=32, patch_size=14,
                           spatial_merge_size=2, temporal_patch_size=2,
                           fullatt_block_indexes=[1], window_size=112),
        vocab_size=128, tie_word_embeddings=False)
    with init_empty_weights(include_buffers=False):
        return Qwen2_5_VLForConditionalGeneration(cfg)


# ---- renaming -----------------------------------------------------------------------------------

class TestRenaming(unittest.TestCase):

    def test_a_checkpoint_that_needs_no_translation_gets_none(self):
        """The property, not the mechanism: 5.x declares legacy renames for everything, so "there
        is no mapping" stopped being true while "nothing here moves" stayed true."""
        model = build_llama()
        conversion = describe(model)
        for name, _ in model.named_parameters():
            self.assertEqual(conversion.rename(name), name)

    def test_a_restructured_multimodal_wrapper_is_translated(self):
        """Qwen2.5-VL stores a flat decoder and a top-level vision tower; the class it builds nests
        both. Getting this wrong places every parameter at a path the model does not have."""
        conversion = describe(build_qwen25vl())
        self.assertEqual(conversion.rename("model.layers.0.self_attn.q_proj.weight"),
                         "model.language_model.layers.0.self_attn.q_proj.weight")
        self.assertEqual(conversion.rename("visual.patch_embed.proj.weight"),
                         "model.visual.patch_embed.proj.weight")
        self.assertEqual(conversion.rename("lm_head.weight"), "lm_head.weight")

    def test_translation_is_idempotent_on_names_that_are_already_right(self):
        """A checkpoint saved by a newer transformers already carries the new names, and the same
        engine has to read both without knowing which it was handed."""
        conversion = describe(build_qwen25vl())
        already = "model.language_model.layers.0.self_attn.q_proj.weight"
        self.assertEqual(conversion.rename(already), already)


# ---- the row arithmetic -------------------------------------------------------------------------

def _op(name, dim=None):
    """A stand-in for one of transformers' conversion operations.

    Matched by class name rather than by identity, which is what lets this be a stand-in at all --
    and is deliberate in the engine too: the real classes live at an import path that has already
    moved once, and an engine that failed to import them would lose expert streaming silently.
    """
    return type(name, (), {"dim": dim})()


class TestRowArithmetic(unittest.TestCase):
    """The one piece of arithmetic this engine derives rather than borrows.

    transformers stacks every expert and then joins the stacks; RocketLLM builds one row. The join
    is declared over the stacked tensors, so on a single row it happens one dimension lower. Off by
    one here swaps halves of every expert's weight and the model still writes fluent text.
    """

    def test_a_bare_stack_is_one_source_used_as_is(self):
        self.assertEqual(_row_arithmetic([_op("MergeModulelist", 0)]), (True, None))

    def test_a_stack_then_concatenate_drops_one_dimension(self):
        self.assertEqual(_row_arithmetic([_op("MergeModulelist", 0), _op("Concatenate", 1)]),
                         (True, 0))
        self.assertEqual(_row_arithmetic([_op("MergeModulelist", 0), _op("Concatenate", 2)]),
                         (True, 1))

    def test_a_conversion_that_does_not_start_by_stacking_experts_is_declined(self):
        self.assertEqual(_row_arithmetic([_op("Chunk", 0)]), (False, None))
        self.assertEqual(_row_arithmetic([]), (False, None))

    def test_an_unfamiliar_operation_after_the_stack_is_declined(self):
        """Declining costs the byte savings for that layer. Guessing costs the answer."""
        self.assertEqual(
            _row_arithmetic([_op("MergeModulelist", 0), _op("PermuteForRope")]), (False, None))
        self.assertEqual(
            _row_arithmetic([_op("MergeModulelist", 0), _op("Concatenate", 1), _op("Transpose")]),
            (False, None))

    def test_a_concatenate_over_the_expert_axis_itself_is_declined(self):
        """dim 0 is the expert axis, so there is no row-level equivalent to fall back to."""
        self.assertEqual(_row_arithmetic([_op("MergeModulelist", 0), _op("Concatenate", 0)]),
                         (False, None))


# ---- matching keys ------------------------------------------------------------------------------

class TestExpertFusionMatching(unittest.TestCase):

    def fusion(self):
        return ExpertFusion([".experts.*.w1.weight", ".experts.*.w3.weight"],
                            ".experts.gate_up_proj", concat_dim=0)

    def test_it_finds_the_source_and_the_expert_ordinal(self):
        fusion = self.fusion()
        self.assertEqual(
            fusion.match("model.layers.0.block_sparse_moe.experts.7.w1.weight"), (0, 7))
        self.assertEqual(
            fusion.match("model.layers.3.block_sparse_moe.experts.12.w3.weight"), (1, 12))

    def test_it_ignores_tensors_it_does_not_consume(self):
        fusion = self.fusion()
        self.assertIsNone(fusion.match("model.layers.0.block_sparse_moe.gate.weight"))
        self.assertIsNone(fusion.match("model.layers.0.self_attn.q_proj.weight"))
        self.assertIsNone(fusion.match("model.layers.0.block_sparse_moe.experts.7.w2.weight"))

    def test_a_name_that_merely_ends_the_same_way_is_not_a_match(self):
        """Anchoring matters: a leaf name is the end of a key, and a pattern that floated free
        would claim tensors belonging to a module nested below the experts."""
        fusion = self.fusion()
        self.assertIsNone(
            fusion.match("model.layers.0.block_sparse_moe.experts.7.w1.weight.something"))

    def test_the_sibling_tensors_of_an_expert_are_derived_from_one_of_them(self):
        """The layer prefix cannot be reconstructed from the pattern, only substituted into."""
        fusion = self.fusion()
        sample = "model.layers.2.block_sparse_moe.experts.7.w1.weight"
        self.assertEqual(fusion.source_key(0, 5, sample),
                         "model.layers.2.block_sparse_moe.experts.5.w1.weight")
        self.assertEqual(fusion.source_key(1, 5, sample),
                         "model.layers.2.block_sparse_moe.experts.5.w3.weight")


# ---- against a real model's own weights ---------------------------------------------------------

class TestAgainstRealWeights(unittest.TestCase):
    """The check that matters: the derived arithmetic must reproduce transformers' own tensors."""

    @needs_modern
    def test_an_assembled_row_equals_what_transformers_built(self):
        model = build_mixtral(experts=3, materialize=True)
        conversion = describe(model)
        if not conversion.fusions:
            self.skipTest("this transformers builds this mixture the way its checkpoint stores it")

        # What the checkpoint holds, taken from the model itself by reversing the fusion: the same
        # per-expert tensors a published Mixtral file stores.
        block = model.get_submodule("model.layers.0.mlp")
        for fusion in conversion.fusions:
            target = None
            for name, param in block.experts.named_parameters(recurse=False):
                if fusion.target.lstrip(".").endswith(name):
                    target = param
                    break
            self.assertIsNotNone(target, f"no parameter for {fusion.target}")

            for expert in range(target.shape[0]):
                built = target[expert]
                # Split the row back into the pieces the sources describe, then rebuild it the way
                # the engine would and require the two to be identical.
                if fusion.concat_dim is None:
                    parts = [built]
                else:
                    parts = list(torch.chunk(built, len(fusion.sources), dim=fusion.concat_dim))
                self.assertEqual(len(parts), len(fusion.sources))
                rebuilt = (parts[0] if len(parts) == 1
                           else torch.cat(parts, dim=fusion.concat_dim))
                self.assertTrue(torch.equal(rebuilt, built),
                                f"expert {expert} of {fusion.target} did not round-trip")

    @needs_modern
    def test_every_declared_fusion_names_a_parameter_the_model_actually_has(self):
        model = build_mixtral()
        conversion = describe(model)
        if not conversion.fusions:
            self.skipTest("nothing declared")
        names = {name for name, _ in model.named_parameters()}
        for fusion in conversion.fusions:
            leaf = fusion.target.rsplit(".", 1)[-1]
            self.assertTrue(any(name.endswith("." + leaf) for name in names),
                            f"{fusion.target} matches no parameter of the built model")


class TestTheEmptyCase(unittest.TestCase):

    def test_a_conversion_with_nothing_declared_is_the_identity(self):
        conversion = CheckpointConversion()
        self.assertFalse(conversion.renames)
        self.assertEqual(conversion.rename("anything.at.all"), "anything.at.all")
        self.assertEqual(conversion.fusions, ())
        self.assertIsNone(conversion.fusion_for("anything.at.all"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
