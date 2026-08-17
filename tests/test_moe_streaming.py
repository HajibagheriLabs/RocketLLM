"""End-to-end tests for MoE expert streaming.

Detection deciding correctly is one thing; the engine then streaming the right bytes and producing
the right numbers is another, and only this file checks the second. Each model is built, saved,
loaded once normally and once through RocketLLM, and the two sets of logits must be *identical* --
not close. Streaming changes when a weight is on the device, never what it holds, so any difference
at all is a bug rather than a tolerance.

Three paths are covered because this change can break any of them independently:

  * a mixture stored as a list of expert modules, which streams one expert at a time;
  * a mixture stored as fused ``[num_experts, ...]`` tensors, which streams the routed rows of one;
  * a dense model, which must be entirely unaffected and has no MoE assertion of its own to fail.

The fused architecture here is synthetic. It is written to the contract every real fused mixture
follows -- experts batched into one tensor per projection, a router sibling, and unrouted experts
scaled to zero -- because the small real ones are awkward to run on CPU, and pinning these
assertions to one vendor's model would test that model rather than the contract.

Savings are asserted as well as correctness, since a path that quietly streamed everything would
still produce the right logits and would still be a total regression of the feature. What can be
asserted differs by layout, and the difference is real rather than an oversight: the fused path
intercepts the router, so a one-token forward provably reads exactly top-k rows. The module-list
path streams whichever expert modules the model chooses to call, and some transformers versions call
every expert and mask the unrouted ones -- so what is pinned there is the guarantee that does hold,
that the experts are loaded one at a time and never all resident together.

Runs on CPU by default. Set ROCKETLLM_TEST_DEVICE to exercise an accelerator's transfer path.
"""
import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from rocketllm import base as base_module
from rocketllm.base import RocketModel
from rocketllm.memory import is_expert
from rocketllm.moe.detect import LAYOUT_FUSED, LAYOUT_FUSED_MERGE, LAYOUT_MODULE_LIST
from rocketllm.moe.expert_cache import is_shared

# The emulated-hardware helpers live with the portability matrix; residency is only observable on
# a device with a budget, which the CPU backend legitimately does not have.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_portability import GB, EmulatedDevice, emulate  # noqa: E402

PROMPT = torch.tensor([[1, 5, 9, 14, 3]])
ONE_TOKEN = torch.tensor([[7]])

#: Layouts under which each expert is a cache entry of its own. Only the module-list layout is:
#: the two fused layouts have no per-expert module to hook, so residency is decided for the rows a
#: token reads rather than for experts held across tokens. Which of them a given checkpoint gets is
#: a property of the transformers release, not of the checkpoint -- a per-expert Mixtral file is a
#: module list on 4.x and a fused module assembled from those same tensors on 5.x -- so the tests
#: that are about per-expert entries ask this rather than assuming an answer.
PER_EXPERT_ENTRY_LAYOUTS = (LAYOUT_MODULE_LIST,)


def sole_container(model):
    """The one expert container of the first layer that has one."""
    layout = next(iter(model._expert_layouts.values()))
    return layout.containers[0]


def requires_per_expert_entries(case, model):
    container = sole_container(model)
    if container.layout not in PER_EXPERT_ENTRY_LAYOUTS:
        case.skipTest(f"this transformers builds the experts as {container.layout}, which has no "
                      f"per-expert module to give its own cache entry")
    return container

#: CPU by default so this file runs on a plain CI runner, which is a property the suite has to keep.
#: Point it at an accelerator to exercise the real transfer path -- async copy streams, a genuine
#: host-to-device copy, and the device-side scatter the fused path does -- none of which the CPU
#: backend reaches. The reference runs on the same device, because the comparison is streaming
#: against no streaming, not one device's kernels against another's.
DEVICE = os.environ.get("ROCKETLLM_TEST_DEVICE", "cpu")


class StreamedModel(RocketModel):
    """RocketModel without the tokenizer, which these checkpoints have no reason to ship."""

    def get_tokenizer(self, hf_token=None):
        return None


# -- a synthetic fused-expert architecture ---------------------------------------------------------

class FusedMoeConfig(PretrainedConfig):
    model_type = "rocketllm_test_fused_moe"

    def __init__(self, hidden_size=16, expert_inner=12, num_hidden_layers=2, num_local_experts=4,
                 num_experts_per_tok=2, vocab_size=32, **kwargs):
        self.hidden_size = hidden_size
        self.expert_inner = expert_inner
        self.num_hidden_layers = num_hidden_layers
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.vocab_size = vocab_size
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


class FusedExperts(nn.Module):
    """Every expert batched into one tensor per projection, run with a single bmm."""

    def __init__(self, config):
        super().__init__()
        experts, hidden, inner = (config.num_local_experts, config.hidden_size,
                                  config.expert_inner)
        self.gate_up_proj = nn.Parameter(torch.empty(experts, hidden, 2 * inner))
        self.down_proj = nn.Parameter(torch.empty(experts, inner, hidden))

    def forward(self, routed):
        gate_up = torch.bmm(routed, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.bmm(up * torch.nn.functional.silu(gate), self.down_proj)


class FusedMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.experts = FusedExperts(config)
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)

    def forward(self, hidden):
        shape = hidden.shape
        flat = hidden.reshape(-1, self.hidden_size)
        logits = self.router(flat)
        top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
        # Unselected experts get -inf, so the sigmoid below makes their scale exactly zero. That is
        # what lets the engine leave their rows unread: a zero-scaled input carries no weight of
        # theirs into the result. Real fused mixtures do the same thing.
        scores = torch.sigmoid(torch.full_like(logits, float("-inf"))
                               .scatter_(1, top_indices, top_values))
        routed = flat.unsqueeze(0).expand(self.num_experts, -1, -1) * scores.t().unsqueeze(-1)
        return self.experts(routed).sum(dim=0).reshape(shape)


class FusedMoeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.attn_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mlp = FusedMoeBlock(config)

    def forward(self, hidden):
        hidden = hidden + self.attn_proj(self.input_layernorm(hidden))
        return hidden + self.mlp(hidden)


class FusedMoePreTrainedModel(PreTrainedModel):
    config_class = FusedMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _supports_sdpa = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.15)
        elif isinstance(module, FusedExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=0.15)
            module.down_proj.data.normal_(mean=0.0, std=0.15)


class FusedMoeModel(FusedMoePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([FusedMoeLayer(config)
                                     for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.post_init()

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class FusedMoeForCausalLM(FusedMoePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = FusedMoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, **kwargs):
        return CausalLMOutputWithPast(logits=self.lm_head(self.model(input_ids)))


def register_fused_architecture():
    """Make the synthetic architecture visible to AutoConfig / AutoModelForCausalLM."""
    try:
        AutoConfig.register(FusedMoeConfig.model_type, FusedMoeConfig)
        AutoModelForCausalLM.register(FusedMoeConfig, FusedMoeForCausalLM)
    except ValueError:
        pass  # already registered by an earlier test in this process


# -- helpers ----------------------------------------------------------------------------------------

def save(model, root):
    model = model.to(torch.float32).eval()
    model.config.torch_dtype = "float32"
    model.save_pretrained(root, safe_serialization=True)
    return model


def mixtral(root):
    from transformers import MixtralConfig, MixtralForCausalLM
    config = MixtralConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                           num_local_experts=4, num_experts_per_tok=2,
                           max_position_embeddings=64, tie_word_embeddings=False)
    return save(MixtralForCausalLM(config), root)


def qwen2_moe(root):
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
    config = Qwen2MoeConfig(hidden_size=32, intermediate_size=64, moe_intermediate_size=32,
                            shared_expert_intermediate_size=48, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                            num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
                            max_position_embeddings=64, tie_word_embeddings=False)
    return save(Qwen2MoeForCausalLM(config), root)


def dense_llama(root):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                         max_position_embeddings=64, tie_word_embeddings=False)
    return save(LlamaForCausalLM(config), root)


def fused_moe(root):
    register_fused_architecture()
    return save(FusedMoeForCausalLM(FusedMoeConfig()), root)


class StreamingCase(unittest.TestCase):
    """Builds one checkpoint per class and reuses it: saving and splitting dominates the runtime."""

    build = None

    @classmethod
    def setUpClass(cls):
        if cls.build is None:
            raise unittest.SkipTest("base class")
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "model"
        cls.root.mkdir(parents=True)
        cls.reference = cls.build(cls.root).to(DEVICE)
        with torch.no_grad():
            cls.expected = cls.reference(PROMPT.to(DEVICE)).logits
            cls.expected_one = cls.reference(ONE_TOKEN.to(DEVICE)).logits

    @classmethod
    def tearDownClass(cls):
        tmp = getattr(cls, "_tmp", None)
        if tmp is not None:
            tmp.cleanup()

    def stream(self):
        return StreamedModel(str(self.root), device=DEVICE, dtype=torch.float32)

    @contextlib.contextmanager
    def stream_with_room(self):
        """Stream on an emulated device with memory to spare, whatever this machine really has.

        Residency is only observable on a device with a budget, and on the CPU backend the derived
        budget is legitimately zero -- there is no separate device pool to hold anything in. A test
        of what the cache KEEPS therefore has nothing to look at here and used to skip, which meant
        the mixture's whole reason for having a residency policy went unchecked on every CI run.
        Describing a roomier machine costs nothing and puts the assertion back.
        """
        device = EmulatedDevice("mixture residency", usable_bytes=1 * GB, window_fraction=0.25,
                                host_cache_bytes=1 * GB)
        with emulate(device):
            model = self.stream()
            try:
                yield model
            finally:
                model.close()

    #: Whether streamed logits must match the reference bit for bit.
    #:
    #: True for every layout that hands the matmul the checkpoint's own tensors: streaming moves
    #: those weights, it does not rebuild them, so any difference at all is a bug.
    #:
    #: The fused layout is the one exception, and it is a property of the optimisation rather than
    #: a defect. Reading only the routed rows means the tensor bound to the module is *allocated
    #: here* -- full width, routed rows filled, the rest zeroed -- instead of being the parameter
    #: the reference model uses. The two hold the same numbers and produce the same result in exact
    #: arithmetic, but they are different allocations, and a CPU GEMM is free to pick a different
    #: kernel and therefore a different summation order for one than the other. That shows up as a
    #: last-place difference of about 6e-08, which is one ULP of float32, and it appears on some
    #: CPUs and not others -- exactly what a reduction-order difference looks like and nothing like
    #: what a wrong weight looks like, which would be wrong by the size of the weight.
    bitwise = True

    def assert_identical(self, model, ids, expected):
        with torch.no_grad():
            got = model(ids.to(DEVICE)).logits
        self.assertEqual(got.shape, expected.shape)
        if self.bitwise:
            self.assertTrue(torch.equal(got, expected),
                            f"streamed logits differ by up to "
                            f"{(got - expected).abs().max().item():.3e}; streaming must not change "
                            f"the arithmetic, only where the weights live")
            return

        # Tight on purpose. float32 epsilon is 1.2e-07, so this admits a few ULP and nothing more:
        # a genuinely wrong expert would be wrong by orders of magnitude, not by rounding.
        torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6,
                                   msg=lambda default: (
                                       f"{default}\nthe fused path may re-associate a sum, but it "
                                       f"may not read a different expert; a difference this large "
                                       f"is the second thing, not the first"))
        # What the difference must never reach: the token the model would actually emit.
        self.assertTrue(torch.equal(got.argmax(dim=-1), expected.argmax(dim=-1)),
                        "the fused path changed which token the model predicts")


class TestModuleListStreaming(StreamingCase):
    build = staticmethod(mixtral)

    def test_logits_are_identical_to_a_full_load(self):
        model = self.stream()
        try:
            self.assert_identical(model, PROMPT, self.expected)
        finally:
            model.close()

    def test_the_layout_is_recognised(self):
        """A per-expert Mixtral checkpoint, streamed per expert -- however this transformers builds it.

        4.x builds an ``nn.ModuleList`` and the checkpoint matches it. 5.x builds one batched module
        over those same per-expert tensors, so the rows have to be assembled as they are read. Both
        are correct readings of one file, and both stream a token's own experts rather than the
        layer's, which is what this asserts. What must never happen is neither.
        """
        model = self.stream()
        try:
            self.assertTrue(model._expert_streaming)
            self.assertEqual(len(model._expert_layouts), 2)
            for layout in model._expert_layouts.values():
                container = layout.containers[0]
                self.assertIn(container.layout, (LAYOUT_MODULE_LIST, LAYOUT_FUSED_MERGE))
                self.assertEqual(container.num_experts, 4)
                self.assertEqual(container.top_k, 2)
        finally:
            model.close()

    def test_experts_are_not_loaded_with_their_layer(self):
        """The saving only exists if the layer's own stream stops carrying the expert weights."""
        model = self.stream()
        try:
            for idx, keys in model._non_expert_keys.items():
                self.assertTrue(keys)
                self.assertFalse([k for k in keys if ".experts." in k],
                                 f"layer {idx} still streams expert tensors with the layer")
        finally:
            model.close()

    def test_experts_are_cached_under_expert_keys(self):
        """An expert filed as a dense entry would get a replacement policy chosen for something else.

        The cache runs LFU-with-aging for experts and FIFO-plus-pinning for dense modules, and the
        asymmetry is deliberate: expert popularity is skewed and predicts itself, while a cyclic
        scan over decoder layers defeats any recency rule. The keys are what select between them.
        """
        model = self.stream()
        try:
            requires_per_expert_entries(self, model)
            with torch.no_grad():
                model(ONE_TOKEN.to(DEVICE))
            keys = [key for key in model._unit_tensor_keys if is_expert(key[1])]
            self.assertTrue(keys)
            self.assertEqual(len(keys), 8, "two layers of four experts should each be an entry")
            self.assertTrue(all(model._unit_byte_counts[key] > 0 for key in keys),
                            "an expert sized at zero bytes is free to keep and never evicted")
        finally:
            model.close()

    def test_only_the_routed_experts_are_read_when_the_rows_have_to_be_assembled(self):
        """The saving has to survive the rearrangement, or the port bought correctness with bytes.

        Where transformers builds one batched expert module over a per-expert checkpoint, a row of
        it is assembled rather than sliced. That is more arithmetic, and it must not be more
        reading: a per-expert file already stores each expert separately, so a top-2 router over
        four experts must ask for two experts' tensors and no others.
        """
        model = self.stream()
        container = sole_container(model)
        if not container.is_merged:
            model.close()
            self.skipTest("this transformers builds the experts the way the checkpoint stores them")

        reads = []
        original = base_module.load_layer_subset

        def recording(path, layer_name, keys):
            keys = list(keys)
            reads.append((layer_name, keys))
            return original(path, layer_name, keys)

        base_module.load_layer_subset = recording
        try:
            with torch.no_grad():
                model(ONE_TOKEN.to(DEVICE))
        finally:
            base_module.load_layer_subset = original
            model.close()

        expert_reads = [keys for _, keys in reads if any(".experts." in key for key in keys)]
        self.assertTrue(expert_reads, "no expert tensors were read at all")
        for keys in expert_reads:
            ordinals = {key.split(".experts.")[1].split(".")[0] for key in keys}
            self.assertEqual(len(ordinals), 2,
                             f"a top-2 router over 4 experts read {len(ordinals)} of them: "
                             f"{sorted(ordinals)}")

    def test_a_repeated_forward_does_not_re_read_its_experts(self):
        """Residency across tokens is the reason experts go through the cache at all.

        Without it a mixture re-reads every expert it touches on every token, which is worse than
        the whole-layer streaming it replaced, because that at least kept the layer resident.
        """
        with self.stream_with_room() as model:
            with torch.no_grad():
                model(ONE_TOKEN.to(DEVICE))
            self.assertGreater(model.cache.device.capacity, 0)
            before = model.cache.report()["fetches"]
            with torch.no_grad():
                model(ONE_TOKEN.to(DEVICE))
            self.assertEqual(model.cache.report()["fetches"], before,
                             "the second token re-read weights the cache already held")

    def test_reset_gives_the_experts_back(self):
        """Residency is bounded by the generation: reset() has to release experts like anything else."""
        model = self.stream()
        try:
            container = sole_container(model)
            with torch.no_grad():
                model(ONE_TOKEN.to(DEVICE))
            model.reset()
            # Reached through the container's own path rather than a hardcoded module name: what
            # the mixture block is called moved between transformers releases (block_sparse_moe ->
            # mlp) and the point of the test is where the weights are, not what the path spells.
            experts = model.layers[1].get_submodule(container.path)
            devices = {p.device.type for p in experts.parameters()}
            self.assertEqual(devices, {"meta"})
        finally:
            model.close()


class TestSharedExperts(StreamingCase):
    """Qwen2-MoE: a mixture with an always-on feed-forward path beside the routed experts.

    A shared expert is read on every token, so it is the first thing that should be resident and
    the last thing that should be ranked against experts by popularity it would trivially win. It
    gets an entry of its own, in its own priority class.
    """

    build = staticmethod(qwen2_moe)

    def test_logits_are_identical_to_a_full_load(self):
        model = self.stream()
        try:
            self.assert_identical(model, PROMPT, self.expected)
        finally:
            model.close()

    def test_shared_experts_are_detected_without_being_named(self):
        model = self.stream()
        try:
            shared = model.experts.shared
            self.assertTrue(shared, "the shared expert beside the routed ones was not found")
            self.assertTrue(all(is_shared(key[1]) for key in shared))
            paths = {key[1] for key in shared}
            self.assertTrue(any("shared_expert" in path for path in paths), paths)
            self.assertTrue(all(size > 0 for size in shared.values()))
        finally:
            model.close()

    def test_the_shared_expert_leaves_the_layer_stream(self):
        """It is its own entry now, so the layer must no longer carry it -- nor the experts."""
        model = self.stream()
        try:
            self.assertTrue(model._non_expert_keys)
            for keys in model._non_expert_keys.values():
                self.assertFalse([k for k in keys if "shared_expert" in k])
                self.assertFalse([k for k in keys if ".experts." in k])
                # The router stays: it has to be resident before it can choose.
                self.assertTrue([k for k in keys if k.endswith("mlp.gate.weight")])
        finally:
            model.close()

    def test_shared_experts_outrank_every_routed_expert(self):
        """A class boundary, not a score: no amount of popularity promotes an expert past one."""
        model = self.stream()
        try:
            requires_per_expert_entries(self, model)
            with torch.no_grad():
                model(PROMPT.to(DEVICE))
            candidates = model._pin_candidates()
            shared = [c for c in candidates if is_shared(c.key[1])]
            routed = [c for c in candidates if is_expert(c.key[1])]
            self.assertTrue(shared and routed)
            self.assertLess(max(c.priority for c in shared),
                            min(c.priority for c in routed))
        finally:
            model.close()

    def test_each_routed_expert_is_loaded_at_most_once_per_forward(self):
        """Re-reading an expert within one forward would be a straight loss over whole-layer."""
        model = self.stream()
        requires_per_expert_entries(self, model)
        loaded = []
        original = model._unit_pre_hook

        def counting(module, args):
            original(module, args)
            loaded.append(id(module))

        for mod in model.model.modules():
            if is_expert(getattr(mod, "_rocketllm_unit", (None, ""))[1]):
                mod._forward_pre_hooks.clear()
                mod.register_forward_pre_hook(counting)
        try:
            with torch.no_grad():
                model(ONE_TOKEN.to(DEVICE))
        finally:
            model.close()

        self.assertTrue(loaded)
        self.assertEqual(len(loaded), len(set(loaded)))
        # How many of the eight actually run is the model's decision, not the engine's: transformers
        # 4.51 walks every expert in the layer, later versions skip the unrouted ones. The engine
        # loads exactly those that run, whichever it is.
        self.assertLessEqual(len(loaded), 8)


class TestFusedStreaming(StreamingCase):
    build = staticmethod(fused_moe)
    # See StreamingCase.bitwise: this layout binds a tensor it allocates rather than the
    # checkpoint's own, so the GEMM may reduce in a different order by a ULP.
    bitwise = False

    def test_logits_are_identical_to_a_full_load(self):
        model = self.stream()
        try:
            self.assert_identical(model, PROMPT, self.expected)
        finally:
            model.close()

    def test_a_single_token_is_identical_too(self):
        """Decode is the case that matters: one token, and only its own experts read."""
        model = self.stream()
        try:
            self.assert_identical(model, ONE_TOKEN, self.expected_one)
        finally:
            model.close()

    def test_the_layout_is_recognised(self):
        model = self.stream()
        try:
            self.assertTrue(model._expert_streaming)
            self.assertEqual(len(model._expert_layouts), 2)
            for layout in model._expert_layouts.values():
                container = layout.containers[0]
                self.assertEqual(container.layout, LAYOUT_FUSED)
                self.assertEqual(container.path, "mlp.experts")
                self.assertEqual(container.router_path, "mlp.router")
                self.assertEqual((container.num_experts, container.top_k), (4, 2))
        finally:
            model.close()

    def test_only_the_routed_rows_are_read(self):
        """The whole point: one expert costs its own bytes, not the layer's."""
        model = self.stream()
        requests = []
        original = base_module.load_layer_rows

        def recording(path, layer_name, rows):
            requests.append({key: tuple(indices) for key, indices in rows.items()})
            return original(path, layer_name, rows)

        base_module.load_layer_rows = recording
        try:
            with torch.no_grad():
                model(ONE_TOKEN.to(DEVICE))
        finally:
            base_module.load_layer_rows = original
            model.close()

        self.assertEqual(len(requests), 2, "one row-read per fused container per forward")
        for request in requests:
            self.assertEqual({key.rsplit(".", 1)[-1] for key in request},
                             {"gate_up_proj", "down_proj"},
                             msg=f"unexpected tensors read: {sorted(request)}")
            for key, indices in request.items():
                self.assertEqual(len(indices), 2, f"{key}: a top-2 router read {len(indices)} of 4")

    def test_the_bound_tensor_is_full_width_with_unrouted_rows_zeroed(self):
        model = self.stream()
        seen = {}

        def inspect(module, args):
            seen["gate_up"] = module.gate_up_proj.detach().clone()

        experts = model.model.model.layers[0].mlp.experts
        experts.register_forward_pre_hook(inspect)  # runs after RocketLLM's, so weights are bound
        try:
            with torch.no_grad():
                model(ONE_TOKEN.to(DEVICE))
        finally:
            model.close()

        gate_up = seen["gate_up"]
        self.assertEqual(tuple(gate_up.shape), (4, 16, 24))
        nonzero_rows = [e for e in range(4) if gate_up[e].abs().sum() > 0]
        self.assertEqual(len(nonzero_rows), 2,
                         "exactly the routed experts should carry weights")

    def test_the_router_is_streamed_with_the_layer(self):
        """It has to be resident before it can choose, so it is never part of the mixture."""
        model = self.stream()
        try:
            for keys in model._non_expert_keys.values():
                self.assertTrue([k for k in keys if k.endswith("mlp.router.weight")])
                self.assertFalse([k for k in keys if "experts." in k])
        finally:
            model.close()

    def test_an_unreadable_router_selection_still_produces_the_right_answer(self):
        """The safe fallback: when the choice cannot be read, every row is read instead.

        Silencing the router hook is how an unfamiliar routing rule looks from here -- the selection
        slot is simply never filled -- and the answer must still be exact, just slower.
        """
        model = self.stream()
        rows_read = []
        original = base_module.load_layer_rows

        def recording(path, layer_name, rows):
            rows_read.append(rows)
            return original(path, layer_name, rows)

        for mod in model.model.modules():
            if getattr(mod, "_rocketllm_router", None) is not None:
                mod._forward_hooks.clear()

        base_module.load_layer_rows = recording
        try:
            self.assert_identical(model, ONE_TOKEN, self.expected_one)
        finally:
            base_module.load_layer_rows = original
            model.close()

        self.assertEqual(rows_read, [], "with no selection the whole container must be read")


class TestDenseIsUnaffected(StreamingCase):
    build = staticmethod(dense_llama)

    def test_logits_are_identical_to_a_full_load(self):
        model = self.stream()
        try:
            self.assert_identical(model, PROMPT, self.expected)
        finally:
            model.close()

    def test_no_expert_streaming_is_installed(self):
        model = self.stream()
        try:
            self.assertFalse(model._expert_streaming)
            self.assertEqual(model._expert_layouts, {})
            self.assertEqual(model._non_expert_keys, {})
        finally:
            model.close()

    def test_layers_still_stream_whole(self):
        model = self.stream()
        try:
            with torch.no_grad():
                model(ONE_TOKEN.to(DEVICE))
            hooked = [m for m in model.model.modules() if hasattr(m, "_rocketllm_unit")]
            self.assertEqual(hooked, [])
        finally:
            model.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
