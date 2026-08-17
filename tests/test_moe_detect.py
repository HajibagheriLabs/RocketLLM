"""Tests for structural expert detection and the two expert-streaming paths.

The thing being protected here is a decision made without configuration. Detection looks at a module
tree and a set of checkpoint shapes and decides whether a layer is a mixture and, if so, how its
experts are stored -- and it has to be right about that for architectures nobody has written yet.
Two failure directions matter and they are not symmetric:

  * Missing a real mixture costs speed. The layer streams whole, exactly as before, and the output
    is unaffected. Bad, recoverable, and visible in the benchmark.
  * Claiming a mixture that is not one, or reading the wrong experts for a token, costs
    correctness. Nothing raises: the model keeps generating, slightly wrong. That is the outcome
    every ambiguous case in the detector resolves away from, and most of these tests exist to pin
    that behaviour rather than the happy path.

So the structural cases below deliberately include things that *look* like experts and must not be
treated as such -- a Sequential of stacked layers, a batched tensor with no router beside it, a
mixture whose routing width the config never states.

The end-to-end tests are the ones that would catch a real regression: a tiny MoE is run through the
engine and its logits compared against the same checkpoint loaded normally, and they must agree
exactly. A dense model is run the same way, because this change can only break one of those two
paths and the dense one has no MoE test to fail.

Everything here runs on CPU with no accelerator, no download and no optional package.
"""
import importlib
import unittest

import torch
import torch.nn as nn

from rocketllm.moe.detect import (LAYOUT_FUSED, LAYOUT_MODULE_LIST, detect_expert_layout,
                                  resolve_top_k, summarize)
from rocketllm.moe.router import RouterSelection


def requires_architecture(module_path, reason):
    """Skip a case whose subject the installed transformers does not ship.

    RocketLLM supports a range of transformers versions, and an architecture that exists in the
    newest one is simply absent from the oldest. That is not a fallback to test for -- the detector
    is structural and never names an architecture -- so the case is skipped where its subject does
    not exist and runs everywhere it does. The synthetic layouts above cover the same contract on
    every version.
    """
    try:
        importlib.import_module(module_path)
    except ImportError:
        return unittest.skip(f"this transformers has no {reason}")
    return lambda test: test


def shapes_of(module, prefix):
    """The checkpoint a module would be saved as: every parameter name and shape."""
    return {f"{prefix}.{name}": tuple(p.shape) for name, p in module.named_parameters()}


# -- building blocks for the structural cases ------------------------------------------------------

class TinyMLP(nn.Module):
    """Stands in for one expert."""

    def __init__(self, hidden=8, inner=6):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)


class ModuleListMoE(nn.Module):
    def __init__(self, hidden=8, experts=4, router=True):
        super().__init__()
        self.experts = nn.ModuleList([TinyMLP(hidden) for _ in range(experts)])
        if router:
            self.gate = nn.Linear(hidden, experts, bias=False)


class FusedExperts(nn.Module):
    def __init__(self, hidden=8, inner=6, experts=4):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(experts, hidden, 2 * inner))
        self.down_proj = nn.Parameter(torch.empty(experts, inner, hidden))


class FusedMoE(nn.Module):
    def __init__(self, hidden=8, experts=4, router=True, extra_router=False):
        super().__init__()
        self.experts = FusedExperts(hidden=hidden, experts=experts)
        if router:
            self.router = nn.Linear(hidden, experts, bias=False)
        if extra_router:
            self.second_gate = nn.Linear(hidden, experts, bias=False)


class DenseLayer(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Sequential(nn.Linear(hidden, 16, bias=False), nn.Linear(16, hidden, bias=False))
        self.norm = nn.LayerNorm(hidden)


class GatedLinearAttention(nn.Module):
    """A depthwise convolution beside a projection that happens to share its leading dimension.

    Modelled on Qwen3.5's ``linear_attn``, and it is not a mixture. The conv's weight is
    ``[expanded, 1, kernel]`` and the projection's is ``[expanded, hidden]``, so both lead with the
    same number and the pair has the shape of "N experts with a router in front of them" -- for a
    real model, N of 8192.
    """

    def __init__(self, hidden=8, expanded=32, kernel=4):
        super().__init__()
        self.conv1d = nn.Conv1d(expanded, expanded, kernel, groups=expanded, bias=False)
        self.in_proj_qkv = nn.Linear(hidden, expanded, bias=False)
        self.out_proj = nn.Linear(expanded, hidden, bias=False)


class Layer(nn.Module):
    """A decoder layer wrapping whatever block is under test."""

    def __init__(self, block, hidden=8):
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden, bias=False)
        self.input_layernorm = nn.LayerNorm(hidden)
        self.mlp = block


class TestStructuralDetection(unittest.TestCase):
    PREFIX = "model.layers.0"

    def detect(self, layer, config=None):
        return detect_expert_layout(layer, shapes_of(layer, self.PREFIX), self.PREFIX, config)

    def test_a_depthwise_convolution_is_not_a_mixture_of_experts(self):
        """The most dangerous false positive this detector can produce.

        A batched leading dimension plus one sibling that shares it is the whole of the fused
        signature, and a gated linear-attention block satisfies both by coincidence. Accepting it
        means streaming top-k rows of a convolution every token needs in full: the rest are left
        zero, the model still writes fluent text, and nothing anywhere says so. A config that
        declares a routing width -- which the real one does, for the mixture in the *other* half of
        the same layer -- removes the last thing that used to stop it.
        """
        layer = Layer(GatedLinearAttention(), hidden=8)
        layout = self.detect(layer, {"num_experts_per_tok": 8, "num_experts": 256})

        self.assertEqual(layout.containers, (),
                         f"a convolution was read as experts: "
                         f"{[c.describe() for c in layout.containers]}")

    def test_module_list_layout_is_found(self):
        layer = Layer(ModuleListMoE(experts=4))
        layout = self.detect(layer, {"num_experts_per_tok": 2})

        self.assertEqual(len(layout.containers), 1)
        container = layout.containers[0]
        self.assertEqual(container.layout, LAYOUT_MODULE_LIST)
        self.assertEqual(container.path, "mlp.experts")
        self.assertEqual(container.num_experts, 4)
        self.assertEqual(container.top_k, 2)
        self.assertEqual(sorted(container.expert_keys), [0, 1, 2, 3])
        for index, keys in container.expert_keys.items():
            self.assertTrue(all(f".experts.{index}." in k for k in keys))

    def test_module_list_keys_are_split_from_the_rest_of_the_layer(self):
        layer = Layer(ModuleListMoE(experts=4))
        layout = self.detect(layer)
        container = layout.containers[0]

        self.assertEqual(len(container.keys), 8)  # 4 experts x 2 projections
        # The router and attention stay with the layer: they are read on every token.
        self.assertIn(f"{self.PREFIX}.mlp.gate.weight", layout.other_keys)
        self.assertIn(f"{self.PREFIX}.self_attn.weight", layout.other_keys)
        self.assertFalse(set(container.keys) & set(layout.other_keys))
        self.assertEqual(len(container.keys) + len(layout.other_keys),
                         len(shapes_of(layer, self.PREFIX)))

    def test_fused_layout_is_found_with_its_router(self):
        layer = Layer(FusedMoE(experts=4))
        layout = self.detect(layer, {"num_experts_per_tok": 2})

        self.assertEqual(len(layout.containers), 1)
        container = layout.containers[0]
        self.assertEqual(container.layout, LAYOUT_FUSED)
        self.assertEqual(container.path, "mlp.experts")
        self.assertEqual(container.num_experts, 4)
        self.assertEqual(container.top_k, 2)
        self.assertEqual(container.router_path, "mlp.router")
        self.assertEqual(set(container.fused_shapes), {
            f"{self.PREFIX}.mlp.experts.gate_up_proj",
            f"{self.PREFIX}.mlp.experts.down_proj",
        })

    def test_fused_shapes_are_the_checkpoint_shapes(self):
        """The destination is allocated from these, so they must be the full expert-major shape."""
        layer = Layer(FusedMoE(experts=4))
        container = self.detect(layer, {"num_experts_per_tok": 2}).containers[0]
        self.assertEqual(container.fused_shapes[f"{self.PREFIX}.mlp.experts.gate_up_proj"],
                         (4, 8, 12))

    def test_dense_layer_has_no_experts(self):
        layout = self.detect(DenseLayer())
        self.assertFalse(layout)
        self.assertEqual(layout.containers, ())
        self.assertEqual(len(layout.other_keys), len(shapes_of(DenseLayer(), self.PREFIX)))

    def test_a_sequential_stack_is_not_a_mixture(self):
        """Integer-named children are not enough: a pipeline's stages are not interchangeable."""
        layout = self.detect(DenseLayer())
        self.assertEqual(layout.containers, ())
        self.assertEqual(layout.skipped, ())  # not even worth reporting

    def test_fused_without_a_router_falls_back(self):
        layer = Layer(FusedMoE(experts=4, router=False))
        layout = self.detect(layer, {"num_experts_per_tok": 2})

        self.assertEqual(layout.containers, ())
        self.assertEqual(len(layout.skipped), 1)
        path, reason = layout.skipped[0]
        self.assertEqual(path, "mlp.experts")
        self.assertIn("unconfirmed", reason)
        # Falling back means the layer still loads every one of its tensors.
        self.assertEqual(len(layout.other_keys), len(shapes_of(layer, self.PREFIX)))

    def test_fused_with_two_router_candidates_is_ambiguous(self):
        layer = Layer(FusedMoE(experts=4, extra_router=True))
        layout = self.detect(layer, {"num_experts_per_tok": 2})

        self.assertEqual(layout.containers, ())
        self.assertIn("unconfirmed", layout.skipped[0][1])

    def test_fused_without_a_declared_top_k_falls_back(self):
        """Reading the wrong rows is silent corruption, so an unstated routing width stops us."""
        layer = Layer(FusedMoE(experts=4))
        layout = self.detect(layer, config={"hidden_size": 8})

        self.assertEqual(layout.containers, ())
        self.assertIn("does not declare how many", layout.skipped[0][1])

    def test_module_list_without_a_top_k_still_streams(self):
        """This layout does not need the routing width: the model calls the experts it chose."""
        layer = Layer(ModuleListMoE(experts=4))
        layout = self.detect(layer, config={"hidden_size": 8})

        self.assertEqual(len(layout.containers), 1)
        self.assertIsNone(layout.containers[0].top_k)

    def test_module_list_missing_from_the_checkpoint_falls_back(self):
        layer = Layer(ModuleListMoE(experts=4))
        shapes = {k: v for k, v in shapes_of(layer, self.PREFIX).items() if ".experts." not in k}
        layout = detect_expert_layout(layer, shapes, self.PREFIX, None)

        self.assertEqual(layout.containers, ())
        self.assertEqual(len(layout.skipped), 1)
        self.assertIn("per-expert tensors for 0", layout.skipped[0][1])

    def test_a_two_dimensional_stack_is_not_a_fused_mixture(self):
        """An embedding table leads with a big dimension too; only a batch of matrices counts."""

        class Stacked(nn.Module):
            def __init__(self):
                super().__init__()
                self.table = nn.Parameter(torch.empty(4, 8))

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = Stacked()
                self.router = nn.Linear(8, 4, bias=False)

        layout = self.detect(Layer(Block()), {"num_experts_per_tok": 2})
        self.assertEqual(layout.containers, ())

    def test_expert_counts_below_the_floor_are_ignored(self):
        layer = Layer(ModuleListMoE(experts=1))
        self.assertEqual(self.detect(layer).containers, ())

    def test_detection_reads_no_tensor_data(self):
        """Shapes come from the safetensors header; a shape-only mapping must be enough."""
        layer = Layer(ModuleListMoE(experts=4))
        shapes = shapes_of(layer, self.PREFIX)
        self.assertTrue(all(isinstance(v, tuple) for v in shapes.values()))
        self.assertEqual(len(detect_expert_layout(layer, shapes, self.PREFIX, None).containers), 1)


class TestTopKResolution(unittest.TestCase):
    def test_found_at_depth(self):
        self.assertEqual(resolve_top_k({"text_config": {"num_experts_per_tok": 6}}, 64), 6)

    def test_alternate_spellings(self):
        self.assertEqual(resolve_top_k({"moe_topk": 3}, 8), 3)
        self.assertEqual(resolve_top_k({"moe_top_k": 4}, 8), 4)

    def test_out_of_range_is_rejected(self):
        """A field that shares a name but not a meaning must not be believed."""
        self.assertIsNone(resolve_top_k({"num_experts_per_tok": 99}, 8))
        self.assertIsNone(resolve_top_k({"num_experts_per_tok": 0}, 8))

    def test_a_bare_top_k_is_never_read(self):
        """`top_k` is a sampling parameter; reading it as a routing width streams wrong experts."""
        self.assertIsNone(resolve_top_k({"top_k": 50}, 8))

    def test_missing_config(self):
        self.assertIsNone(resolve_top_k(None, 8))
        self.assertIsNone(resolve_top_k({}, 8))


class TestRouterSelection(unittest.TestCase):
    def test_logits_become_the_top_k(self):
        selection = RouterSelection(num_experts=4, top_k=2)
        selection.observe(torch.tensor([[0.1, 5.0, 0.2, 3.0]]))
        self.assertEqual(selection.take(), (1, 3))

    def test_the_union_over_several_tokens(self):
        selection = RouterSelection(num_experts=4, top_k=1)
        selection.observe(torch.tensor([[9.0, 0., 0., 0.], [0., 0., 0., 9.0]]))
        self.assertEqual(selection.take(), (0, 3))

    def test_a_monotonic_transform_does_not_change_the_choice(self):
        """Routers differ in whether they softmax before the top-k; the answer must not."""
        logits = torch.tensor([[0.1, 5.0, 0.2, 3.0]])
        plain, softmaxed = RouterSelection(4, 2), RouterSelection(4, 2)
        plain.observe(logits)
        softmaxed.observe(torch.softmax(logits, dim=-1))
        self.assertEqual(plain.take(), softmaxed.take())

    def test_scores_inside_a_tuple_are_found(self):
        selection = RouterSelection(num_experts=4, top_k=1)
        selection.observe((torch.tensor([[0.0, 0.0, 7.0, 0.0]]), None))
        self.assertEqual(selection.take(), (2,))

    def test_an_unreadable_output_is_unknown_rather_than_a_guess(self):
        selection = RouterSelection(num_experts=4, top_k=2)
        self.assertIsNone(selection.observe(torch.tensor([[1, 2]])))  # integer, not scores
        self.assertIsNone(selection.take())

    def test_two_candidate_score_tensors_are_unknown(self):
        selection = RouterSelection(num_experts=4, top_k=4)
        selection.observe((torch.rand(2, 4), torch.rand(2, 4)))
        self.assertIsNone(selection.take())

    def test_taking_clears_the_slot(self):
        """A stale selection would stream the previous token's experts for this one."""
        selection = RouterSelection(num_experts=4, top_k=1)
        selection.observe(torch.tensor([[0.0, 9.0, 0.0, 0.0]]))
        self.assertEqual(selection.take(), (1,))
        self.assertIsNone(selection.take())


class TestSummary(unittest.TestCase):
    def test_identical_layers_collapse_to_one_line(self):
        layer = Layer(ModuleListMoE(experts=4))
        prefix = "model.layers.0"
        layout = detect_expert_layout(layer, shapes_of(layer, prefix), prefix,
                                      {"num_experts_per_tok": 2})
        lines = summarize({1: layout, 2: layout, 3: layout})
        self.assertEqual(len(lines), 1)
        self.assertIn("3 layers", lines[0])
        self.assertIn("4 experts", lines[0])
        self.assertIn("top-k 2", lines[0])

    def test_an_unknown_top_k_is_reported_as_unknown(self):
        layer = Layer(ModuleListMoE(experts=4))
        prefix = "model.layers.0"
        layout = detect_expert_layout(layer, shapes_of(layer, prefix), prefix, None)
        self.assertIn("top-k unknown", summarize({1: layout})[0])


# -- the architectures transformers actually ships -------------------------------------------------

class TestRealArchitectures(unittest.TestCase):
    """Detection has to work on real module trees, not just the ones written for the test.

    These are structural checks only -- no forward pass -- so they stay fast and do not depend on a
    given transformers release getting a model's forward right on CPU.
    """

    def _layer_shapes(self, model, layer_name):
        return {name: tuple(p.shape) for name, p in model.named_parameters()
                if name.startswith(layer_name + ".")}

    def _detect(self, model, config, layer_name="model.layers.0"):
        layer = model.get_submodule(layer_name)
        return detect_expert_layout(layer, self._layer_shapes(model, layer_name), layer_name, config)

    def test_mixtral_experts_are_found_whichever_way_transformers_builds_them(self):
        """The shard here mirrors the model's own parameters, so the two agree by construction.

        Which layout that is, is the running transformers' choice and not the checkpoint's: 4.x
        builds an ``nn.ModuleList`` and 5.x one batched module. Both are found, both name the same
        module path, and both agree on how many experts there are and how many a token visits --
        which is everything the engine goes on to use.
        """
        from transformers import MixtralConfig, MixtralForCausalLM
        config = MixtralConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                               num_local_experts=4, num_experts_per_tok=2,
                               max_position_embeddings=32)
        layout = self._detect(MixtralForCausalLM(config), config)

        self.assertEqual(len(layout.containers), 1)
        container = layout.containers[0]
        self.assertIn(container.layout, (LAYOUT_MODULE_LIST, LAYOUT_FUSED))
        self.assertTrue(container.path.endswith(".experts"), container.path)
        self.assertEqual((container.num_experts, container.top_k), (4, 2))

    def test_qwen2_moe_shared_expert_is_found_beside_the_routed_ones(self):
        """A shared expert runs for every token, so it is pinned rather than streamed or routed."""
        from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
        config = Qwen2MoeConfig(hidden_size=16, intermediate_size=32, moe_intermediate_size=16,
                                shared_expert_intermediate_size=24, num_hidden_layers=1,
                                num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                                num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
                                max_position_embeddings=32)
        layout = self._detect(Qwen2MoeForCausalLM(config), config)

        self.assertEqual(len(layout.containers), 1)
        container = layout.containers[0]
        self.assertEqual(container.path, "mlp.experts")
        self.assertEqual(container.router_path, "mlp.gate")
        self.assertTrue(any("shared_expert" in path for path in container.shared_keys),
                        f"shared expert not detected: {sorted(container.shared_keys)}")
        # The router stays with the layer: it must be resident before it can choose.
        self.assertTrue(any("mlp.gate.weight" in k for k in layout.other_keys))
        self.assertFalse([k for k in layout.other_keys if "shared_expert" in k],
                         "a shared expert is its own entry, not part of the layer's stream")

    @requires_architecture("transformers.models.llama4", "Llama 4 (added in transformers 4.51)")
    def test_llama4_is_fused(self):
        from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
        from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM
        config = Llama4TextConfig(hidden_size=16, intermediate_size=32, intermediate_size_mlp=32,
                                  num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                                  vocab_size=64, num_local_experts=4, num_experts_per_tok=1,
                                  max_position_embeddings=32, interleave_moe_layer_step=1)
        layout = self._detect(Llama4ForCausalLM(config), config)

        self.assertEqual(len(layout.containers), 1)
        container = layout.containers[0]
        self.assertEqual(container.layout, LAYOUT_FUSED)
        self.assertEqual(container.path, "feed_forward.experts")
        self.assertEqual(container.router_path, "feed_forward.router")
        self.assertEqual((container.num_experts, container.top_k), (4, 1))
        # The fused layout has a shared expert too, and it is found the same way.
        self.assertTrue(any("shared_expert" in path for path in container.shared_keys),
                        f"shared expert not detected: {sorted(container.shared_keys)}")

    def test_a_dense_llama_has_nothing_to_find(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                             max_position_embeddings=32)
        layout = self._detect(LlamaForCausalLM(config), config)

        self.assertFalse(layout)
        self.assertEqual(layout.skipped, ())


if __name__ == "__main__":
    unittest.main(verbosity=2)
