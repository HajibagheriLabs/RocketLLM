"""Tests for the pre-quantized checkpoint intake, with the machine mocked out.

Two things are checked here that nothing else can check on one developer's hardware.

The first is size accounting. Every placement and eviction decision downstream is made in packed
bytes, so a format whose logical weight is mis-grouped -- a scale counted as a weight of its own, a
4-bit payload sized as if it were 16-bit -- corrupts the arithmetic for the whole engine and does it
silently, because the model still runs. So each format gets a synthetic checkpoint with hand-checked
byte counts.

The second is that the packed-vs-scratch decision follows the *device* and not the file. The same
synthetic checkpoint is run past two mocked machines, one with the relevant capability and one
without, and the two are required to disagree. That is the property no single machine can
demonstrate, and it is the one that breaks first when a format-specific shortcut creeps in.

Nothing here needs an accelerator, a checkpoint download, or any optional reader package.
"""
import logging
import unittest

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device

from rocketllm.hw import caps as C
from rocketllm.quant import PackedWeight, QuantBackend, TensorSpec, decision_table, detect_backend
from rocketllm.quant.registry import quant_method_of
from rocketllm.quant.safetensors_quant import (BitsAndBytesBackend, CompressedTensorsBackend,
                                               GptqAwqBackend, HfQuantizerBackend,
                                               announce_backend)
from rocketllm.utils import reject_compression_argument

LINEAR = "model.layers.0.self_attn.q_proj"
#: One 512x512 Linear, which is 262144 values -- the number every count below is derived from.
IN_FEATURES = OUT_FEATURES = 512
VALUES = IN_FEATURES * OUT_FEATURES


class FusedPlan:
    """Stand-in for hw.caps.FusedPlan, which is all the backends read of it."""

    def __init__(self, fused):
        self.fused = fused
        self.reason = "mocked kernel inventory"


class FakeCaps:
    """A machine with exactly the capabilities a test asks for, and no others.

    Everything defaults to absent, so a test that forgets to opt in gets the degraded path rather
    than whatever the box running the suite happens to support. `providers` names the fused kernels
    that import and run here, which is not the same question as whether *a* fused kernel exists.
    """

    ALL_PROVIDERS = ("torch_int4pack", "bitsandbytes", "gptqmodel", "marlin_kernels", "exllamav2",
                     "awq_ext")

    def __init__(self, fp4=False, fp8=False, fused_4bit=False, providers=None,
                 compute_dtype=torch.bfloat16):
        self.supports_fp4 = fp4
        self.supports_fp8 = fp8
        self.compute_dtype = compute_dtype
        self._providers = tuple(providers) if providers is not None else (
            self.ALL_PROVIDERS if fused_4bit else ())
        self._fused = bool(self._providers)

    def fused_4bit_plan(self):
        return FusedPlan(self._fused)

    def fused_4bit_providers(self):
        return {name: name in self._providers for name in self.ALL_PROVIDERS}


NOTHING = dict(fp4=False, fp8=False, fused_4bit=False)
EVERYTHING = dict(fp4=True, fp8=True, fused_4bit=True)


# -------------------------------------------------------------------------------------------------
# synthetic checkpoints
#
# Shapes are the real ones each format writes, so the packing factors under test are the packing
# factors in the wild. Contents are zeros: nothing here decodes a payload, it only measures and
# routes one.
# -------------------------------------------------------------------------------------------------

def gptq_shard(bias=True):
    """int32-packed payload, group scales and zero points, and the reorder index."""
    shard = {
        f"{LINEAR}.qweight": torch.zeros(IN_FEATURES // 8, OUT_FEATURES, dtype=torch.int32),
        f"{LINEAR}.qzeros": torch.zeros(IN_FEATURES // 128, OUT_FEATURES // 8, dtype=torch.int32),
        f"{LINEAR}.scales": torch.zeros(IN_FEATURES // 128, OUT_FEATURES, dtype=torch.float16),
        f"{LINEAR}.g_idx": torch.zeros(IN_FEATURES, dtype=torch.int32),
    }
    if bias:
        shard[f"{LINEAR}.bias"] = torch.zeros(OUT_FEATURES, dtype=torch.float16)
    return shard


GPTQ_CONFIG = {"quant_method": "gptq", "bits": 4, "group_size": 128}
AWQ_CONFIG = {"quant_method": "awq", "bits": 4}


def compressed_tensors_shard(with_shape=True):
    """Two 4-bit values per stored byte, plus the scales and the shape that was lost to packing."""
    shard = {
        f"{LINEAR}.weight_packed": torch.zeros(OUT_FEATURES, IN_FEATURES // 2, dtype=torch.uint8),
        f"{LINEAR}.weight_scale": torch.zeros(OUT_FEATURES, IN_FEATURES // 32, dtype=torch.uint8),
    }
    if with_shape:
        shard[f"{LINEAR}.weight_shape"] = torch.tensor([OUT_FEATURES, IN_FEATURES])
    return shard


CT_CONFIG = {"quant_method": "compressed-tensors",
             "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float"}}}}


def bnb_shard(marker=".4bit."):
    """A byte-packed payload with its quant state stored underneath the weight's own name."""
    return {
        f"{LINEAR}.weight": torch.zeros(VALUES // 2, 1, dtype=torch.uint8),
        f"{LINEAR}.weight{marker}absmax": torch.zeros(VALUES // 64, dtype=torch.float32),
        f"{LINEAR}.weight{marker}quant_map": torch.zeros(16, dtype=torch.float32),
    }


BNB_CONFIG = {"quant_method": "bitsandbytes_4bit"}


def fp8_shard():
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None:
        raise unittest.SkipTest("this torch build has no fp8 dtype")
    return {
        f"{LINEAR}.weight": torch.zeros(OUT_FEATURES, IN_FEATURES, dtype=fp8),
        f"{LINEAR}.weight_scale_inv": torch.zeros(4, 4, dtype=torch.float32),
    }


def dense_shard():
    return {
        f"{LINEAR}.weight": torch.zeros(OUT_FEATURES, IN_FEATURES, dtype=torch.bfloat16),
        f"{LINEAR}.bias": torch.zeros(OUT_FEATURES, dtype=torch.bfloat16),
    }


def backend_for(config, **capabilities):
    caps = FakeCaps(**capabilities) if capabilities else FakeCaps()
    return detect_backend(config, caps=caps, compute_dtype=torch.bfloat16)


def weight_named(weights, name):
    return next(w for w in weights if w.name == name)


def only_weight(backend, shard, name=f"{LINEAR}.weight"):
    return weight_named(backend.plan(shard), name)


# -------------------------------------------------------------------------------------------------


class TestBackendSelection(unittest.TestCase):
    """A checkpoint is routed by what it declares about itself, never by a model name."""

    def test_each_declared_method_selects_its_backend(self):
        cases = {
            "gptq": GptqAwqBackend,
            "awq": GptqAwqBackend,
            "autoawq": GptqAwqBackend,
            "compressed-tensors": CompressedTensorsBackend,
            "compressed_tensors": CompressedTensorsBackend,
            "mxfp4": CompressedTensorsBackend,
            "bitsandbytes_4bit": BitsAndBytesBackend,
            "bitsandbytes_8bit": BitsAndBytesBackend,
        }
        for method, expected in cases.items():
            with self.subTest(method=method):
                backend = backend_for({"quant_method": method})
                self.assertIsInstance(backend, expected)

    def test_a_checkpoint_declaring_nothing_is_unquantized(self):
        backend = backend_for(None)
        self.assertIs(type(backend), QuantBackend)
        self.assertEqual(backend.format, "dense")

    def test_an_unknown_method_delegates_rather_than_refusing(self):
        """transformers may know a format this build has never heard of; that must still load."""
        backend = backend_for({"quant_method": "some-future-format"})
        self.assertIsInstance(backend, HfQuantizerBackend)
        self.assertEqual(backend.format, "some-future-format")

    def test_the_method_is_read_through_an_enum_or_an_object(self):
        """transformers models quant_method as an enum on some versions and a string on others."""

        class Method:
            value = "GPTQ"

        class Config:
            quant_method = Method()
            bits = 4

        self.assertEqual(quant_method_of(Config()), "gptq")
        self.assertIsInstance(backend_for(Config()), GptqAwqBackend)

    def test_the_reported_format_is_what_the_checkpoint_calls_itself(self):
        """One class handles GPTQ and AWQ; a bug report should still say which file it was."""
        self.assertEqual(backend_for(GPTQ_CONFIG).format, "gptq")
        self.assertEqual(backend_for(AWQ_CONFIG).format, "awq")

    def test_the_declared_width_is_found_however_deeply_it_is_nested(self):
        """compressed-tensors records the width per config group, not at the top level."""
        self.assertEqual(backend_for(CT_CONFIG).declared_bits(), 4)
        self.assertEqual(backend_for(GPTQ_CONFIG).declared_bits(), 4)

    def test_bitsandbytes_takes_its_width_from_the_method_name(self):
        """Its config spells the width into the method rather than into a field of its own."""
        self.assertEqual(backend_for({"quant_method": "bitsandbytes_4bit"}).declared_bits(), 4)
        self.assertEqual(backend_for({"quant_method": "bitsandbytes_8bit"}).declared_bits(), 8)


class TestLayout(unittest.TestCase):
    """Which checkpoint tensors make up one logical weight."""

    def test_gptq_gathers_its_four_tensors_into_one_weight(self):
        weights = backend_for(GPTQ_CONFIG).plan(gptq_shard())
        weight = weight_named(weights, f"{LINEAR}.weight")
        self.assertEqual(set(weight.tensor_names), {
            f"{LINEAR}.qweight", f"{LINEAR}.qzeros", f"{LINEAR}.scales", f"{LINEAR}.g_idx"})
        self.assertEqual(weight.payload.name, f"{LINEAR}.qweight",
                         "the payload is the weight itself, not whichever tensor came first")

    def test_a_bias_beside_a_quantized_weight_stays_its_own_weight(self):
        weights = backend_for(GPTQ_CONFIG).plan(gptq_shard())
        self.assertEqual({w.name for w in weights}, {f"{LINEAR}.weight", f"{LINEAR}.bias"})

    def test_compressed_tensors_gathers_payload_scale_and_shape(self):
        weight = only_weight(backend_for(CT_CONFIG), compressed_tensors_shard())
        self.assertEqual(len(weight.specs), 3)
        self.assertEqual(weight.payload.name, f"{LINEAR}.weight_packed")

    def test_an_fp8_weight_and_its_block_scale_are_one_weight(self):
        weight = only_weight(backend_for({"quant_method": "fp8"}), fp8_shard())
        self.assertEqual(set(weight.tensor_names),
                         {f"{LINEAR}.weight", f"{LINEAR}.weight_scale_inv"})
        self.assertEqual(weight.payload.name, f"{LINEAR}.weight")

    def test_both_spellings_of_a_bitsandbytes_quant_state_group_the_same(self):
        """RocketLLM's own shards mark it with .4bit.; a HF checkpoint just nests it."""
        for marker in (".4bit.", ".8bit.", "."):
            with self.subTest(marker=marker):
                weight = only_weight(backend_for(BNB_CONFIG), bnb_shard(marker))
                self.assertEqual(len(weight.specs), 3)
                self.assertEqual(weight.payload.name, f"{LINEAR}.weight")

    def test_a_quant_state_is_spanned_but_never_placed(self):
        """The module has no parameter for it; the quantizer reads it back out of the shard."""
        weight = only_weight(backend_for(BNB_CONFIG), bnb_shard())
        self.assertEqual(len(weight.tensor_names), 3)
        self.assertEqual(weight.placed_names, [f"{LINEAR}.weight"])

    def test_an_ordinary_companion_is_placed_on_its_own(self):
        """An fp8 block scale is a real parameter of the module, unlike a bnb quant state."""
        weight = only_weight(backend_for({"quant_method": "fp8"}), fp8_shard())
        self.assertEqual(len(weight.placed_names), 2)

    def test_an_unquantized_shard_is_one_weight_per_tensor(self):
        weights = backend_for(None).plan(dense_shard())
        self.assertEqual([w.name for w in weights], [f"{LINEAR}.weight", f"{LINEAR}.bias"])
        self.assertTrue(all(len(w.specs) == 1 for w in weights))


class TestSizeAccounting(unittest.TestCase):
    """Packed bytes are what the cache places and what crosses the link; they must be exact."""

    def test_gptq_packs_eight_four_bit_values_into_each_int32(self):
        weight = only_weight(backend_for(GPTQ_CONFIG), gptq_shard())
        self.assertEqual(weight.bits, 4)
        # 64x512 int32 payload + 4x64 int32 zeros + 4x512 fp16 scales + 512 int32 indices
        self.assertEqual(weight.packed_bytes, 131072 + 1024 + 4096 + 2048)
        self.assertEqual(weight.expanded_bytes, VALUES * 2)
        self.assertLess(weight.packed_bytes, weight.expanded_bytes)

    def test_compressed_tensors_packs_two_four_bit_values_into_each_byte(self):
        weight = only_weight(backend_for(CT_CONFIG), compressed_tensors_shard())
        self.assertEqual(weight.bits, 4)
        self.assertEqual(weight.packed_bytes, 131072 + 8192 + 2 * 8)
        self.assertEqual(weight.expanded_bytes, VALUES * 2)

    def test_a_shipped_shape_is_preferred_over_the_packing_factor(self):
        """compressed-tensors ships weight_shape because packing destroys the real shape."""
        backend = backend_for(CT_CONFIG)
        with_shape = only_weight(backend, compressed_tensors_shard(with_shape=True))
        without = only_weight(backend, compressed_tensors_shard(with_shape=False))
        self.assertEqual(with_shape.logical_shape, (OUT_FEATURES, IN_FEATURES))
        self.assertIsNone(without.logical_shape)
        # Either way the count has to come out the same; the shape is a cross-check, not a fudge.
        self.assertEqual(with_shape.expanded_bytes, without.expanded_bytes)

    def test_bitsandbytes_counts_the_quant_state_it_ships_with(self):
        weight = only_weight(backend_for(BNB_CONFIG), bnb_shard())
        # 131072-byte payload + 4096 absmax values as fp32 + a 16-entry fp32 map
        self.assertEqual(weight.packed_bytes, 131072 + 16384 + 64)
        self.assertEqual(weight.expanded_bytes, VALUES * 2)

    def test_an_fp8_weight_expands_to_the_compute_dtype(self):
        weight = only_weight(backend_for({"quant_method": "fp8"}), fp8_shard())
        self.assertEqual(weight.bits, 8)
        self.assertEqual(weight.packed_bytes, VALUES + 64)
        self.assertEqual(weight.expanded_bytes, VALUES * 2)

    def test_an_unquantized_weight_costs_the_same_either_way(self):
        weight = only_weight(backend_for(None), dense_shard())
        self.assertEqual(weight.packed_bytes, weight.expanded_bytes)
        self.assertEqual(weight.packed_bytes, VALUES * 2)

    def test_a_bias_is_not_sized_as_if_it_were_four_bit(self):
        """The declared width applies to the quantized weights, not to everything in the file."""
        weights = backend_for(GPTQ_CONFIG).plan(gptq_shard())
        bias = weight_named(weights, f"{LINEAR}.bias")
        self.assertEqual(bias.bits, 16)
        self.assertEqual(bias.packed_bytes, OUT_FEATURES * 2)
        self.assertEqual(bias.expanded_bytes, OUT_FEATURES * 2)

    def test_the_compute_dtype_is_what_expanded_bytes_expands_into(self):
        """A device that lands in fp32 pays twice as much scratch for the same checkpoint."""
        wide = detect_backend(CT_CONFIG, caps=FakeCaps(), compute_dtype=torch.float32)
        self.assertEqual(only_weight(wide, compressed_tensors_shard()).expanded_bytes, VALUES * 4)

    def test_sizes_can_be_had_from_metadata_without_reading_the_tensors(self):
        """The cache has to size a layer before it decides whether to keep it."""
        shard = compressed_tensors_shard()
        specs = [TensorSpec.of(name, tensor) for name, tensor in shard.items()]
        backend = backend_for(CT_CONFIG)
        from_data = only_weight(backend, shard)
        from_header = weight_named(backend.plan_from_specs(specs), f"{LINEAR}.weight")
        self.assertEqual(from_header.packed_bytes, from_data.packed_bytes)
        self.assertEqual(from_header.expanded_bytes, from_data.expanded_bytes)

    def test_a_weight_planned_from_metadata_says_so_rather_than_failing_obscurely(self):
        specs = [TensorSpec.of(n, t) for n, t in dense_shard().items()]
        weight = weight_named(backend_for(None).plan_from_specs(specs), f"{LINEAR}.weight")
        self.assertFalse(weight.has_values)
        with self.assertRaises(ValueError) as raised:
            weight.materialize("cpu")
        self.assertIn("metadata", str(raised.exception))


class TestNeedsScratchFollowsTheDevice(unittest.TestCase):
    """The same checkpoint, two machines, two correct and different answers."""

    def assert_flips(self, config, shard, capable, incapable, name=f"{LINEAR}.weight"):
        packed = only_weight(detect_backend(config, caps=FakeCaps(**capable),
                                            compute_dtype=torch.bfloat16), shard, name)
        scratch = only_weight(detect_backend(config, caps=FakeCaps(**incapable),
                                             compute_dtype=torch.bfloat16), shard, name)
        self.assertFalse(packed.needs_scratch, "a capable device should compute on packed weights")
        self.assertTrue(scratch.needs_scratch, "an incapable device has to expand them first")
        self.assertEqual(packed.scratch_bytes, 0)
        self.assertEqual(scratch.scratch_bytes, scratch.expanded_bytes)

    def test_gptq_follows_the_fused_kernel(self):
        self.assert_flips(GPTQ_CONFIG, gptq_shard(), dict(fused_4bit=True), dict(fused_4bit=False))

    def test_awq_follows_the_fused_kernel(self):
        self.assert_flips(AWQ_CONFIG, gptq_shard(), dict(fused_4bit=True), dict(fused_4bit=False))

    def test_compressed_tensors_follows_native_fp4_not_the_kernel_inventory(self):
        """It ships no compute kernels, so installing one cannot rescue it -- only the device can."""
        self.assert_flips(CT_CONFIG, compressed_tensors_shard(), dict(fp4=True), dict(fp4=False))
        with_kernel = only_weight(detect_backend(CT_CONFIG, caps=FakeCaps(fused_4bit=True),
                                                 compute_dtype=torch.bfloat16),
                                  compressed_tensors_shard())
        self.assertTrue(with_kernel.needs_scratch,
                        "a fused kernel package is irrelevant to compressed-tensors")

    def test_bitsandbytes_follows_the_fused_kernel_because_it_is_the_kernel(self):
        self.assert_flips(BNB_CONFIG, bnb_shard(), dict(fused_4bit=True), dict(fused_4bit=False))

    def test_a_kernel_that_cannot_read_this_packing_does_not_count(self):
        """Every format packs its own way; only a kernel written for it can skip the expansion."""
        cases = [
            (BNB_CONFIG, bnb_shard(), ["torch_int4pack"], True),
            (BNB_CONFIG, bnb_shard(), ["bitsandbytes"], False),
            (GPTQ_CONFIG, gptq_shard(), ["bitsandbytes"], True),
            (GPTQ_CONFIG, gptq_shard(), ["gptqmodel"], False),
        ]
        for config, shard, providers, expected in cases:
            backend = detect_backend(config, caps=FakeCaps(providers=providers),
                                     compute_dtype=torch.bfloat16)
            with self.subTest(fmt=config["quant_method"], providers=providers):
                self.assertIs(only_weight(backend, shard).needs_scratch, expected)

    def test_fp8_follows_fp8_arithmetic(self):
        self.assert_flips({"quant_method": "fp8"}, fp8_shard(), dict(fp8=True), dict(fp8=False))

    def test_an_unquantized_weight_never_needs_scratch(self):
        for capabilities in (NOTHING, EVERYTHING):
            backend = detect_backend(None, caps=FakeCaps(**capabilities),
                                     compute_dtype=torch.bfloat16)
            with self.subTest(capabilities=capabilities):
                self.assertFalse(only_weight(backend, dense_shard()).needs_scratch)

    def test_a_sixteen_bit_tensor_in_a_quantized_checkpoint_never_needs_scratch(self):
        weights = detect_backend(GPTQ_CONFIG, caps=FakeCaps(**NOTHING),
                                 compute_dtype=torch.bfloat16).plan(gptq_shard())
        self.assertFalse(weight_named(weights, f"{LINEAR}.bias").needs_scratch)

    def test_the_capability_is_queried_when_asked_not_cached_from_construction(self):
        """A backend built before the device was probed must not answer from a stale reading."""
        caps = FakeCaps(fused_4bit=False)
        backend = detect_backend(GPTQ_CONFIG, caps=caps, compute_dtype=torch.bfloat16)
        weight = only_weight(backend, gptq_shard())
        self.assertTrue(weight.needs_scratch)
        caps._providers = ("gptqmodel",)
        self.assertFalse(weight.needs_scratch)


class TestPlacement(unittest.TestCase):
    """What actually reaches the module, on a real (if tiny) model on the CPU."""

    def build(self, config, extra_params=(), quantizer=None, **capabilities):
        model = nn.Module()
        linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True, dtype=torch.bfloat16)
        for name, shape, dtype in extra_params:
            linear.register_parameter(
                name, nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False))
        # Park it on meta, as the engine does between layers.
        model.add_module("model", nn.Module())
        model.model.add_module("layers", nn.ModuleList([nn.Module()]))
        model.model.layers[0].add_module("self_attn", nn.Module())
        model.model.layers[0].self_attn.add_module("q_proj", linear)
        for param_name, _ in list(model.named_parameters()):
            set_module_tensor_to_device(model, param_name, "meta")

        backend = detect_backend(config, caps=FakeCaps(**capabilities), model=model,
                                 hf_quantizer=quantizer, compute_dtype=torch.bfloat16,
                                 device="cpu")
        return model, backend

    def test_a_plain_weight_is_cast_to_the_compute_dtype(self):
        model, backend = self.build(None)
        shard = {f"{LINEAR}.weight": torch.zeros(OUT_FEATURES, IN_FEATURES, dtype=torch.float32)}
        placed = only_weight(backend, shard).materialize("cpu")
        self.assertEqual(placed, [f"{LINEAR}.weight"])
        self.assertIs(model.model.layers[0].self_attn.q_proj.weight.dtype, torch.bfloat16)

    def test_a_packed_payload_is_placed_without_a_cast(self):
        """Casting packed integers to a float dtype turns the weight into noise."""
        packed_shape = (OUT_FEATURES, IN_FEATURES // 2)
        model, backend = self.build(
            CT_CONFIG, extra_params=[("weight_packed", packed_shape, torch.uint8)])
        packed = torch.zeros(packed_shape, dtype=torch.uint8)
        backend.place(f"{LINEAR}.weight_packed", packed, "cpu")
        self.assertIs(model.model.layers[0].self_attn.q_proj.weight_packed.dtype, torch.uint8)

    def test_an_eight_bit_float_is_placed_without_a_cast(self):
        fp8 = getattr(torch, "float8_e4m3fn", None)
        if fp8 is None:
            self.skipTest("this torch build has no fp8 dtype")
        backend = backend_for({"quant_method": "fp8"})
        self.assertTrue(backend.load_verbatim(f"{LINEAR}.weight",
                                              torch.zeros(4, 4, dtype=fp8)))

    def test_a_companion_keeps_its_own_dtype(self):
        backend = backend_for(CT_CONFIG)
        scale = torch.zeros(4, 4, dtype=torch.float32)
        self.assertIs(backend.target_dtype(f"{LINEAR}.weight_scale", scale), torch.float32)
        plain = torch.zeros(4, 4, dtype=torch.float32)
        self.assertIs(backend.target_dtype(f"{LINEAR}.weight", plain), torch.bfloat16)

    def test_the_quantizer_reconstructs_the_params_it_claims(self):
        calls = []

        class Quantizer:
            def param_needs_quantization(self, model, name):
                return name.endswith(".weight")

            def create_quantized_param(self, model, value, name, device, state_dict):
                calls.append((name, tuple(sorted(state_dict))))

        model, backend = self.build(BNB_CONFIG, quantizer=Quantizer())
        shard = bnb_shard()
        weight = only_weight(backend, shard)
        self.assertEqual(weight.placements(), [], "nothing here is an ordinary placement")
        self.assertEqual(weight.quantizer_names(), [f"{LINEAR}.weight"])

        weight.materialize("cpu", shard=shard)
        self.assertEqual(len(calls), 1)
        name, seen = calls[0]
        self.assertEqual(name, f"{LINEAR}.weight")
        self.assertEqual(len(seen), 3, "the quantizer must see the quant state, not just the payload")

    def test_the_older_transformers_spelling_of_the_query_still_works(self):
        """transformers renamed check_quantized_param -> param_needs_quantization."""
        seen = []

        class OldQuantizer:
            def check_quantized_param(self, model, param_value, param_name, state_dict):
                seen.append(param_name)
                return False

        _, backend = self.build(GPTQ_CONFIG, quantizer=OldQuantizer())
        self.assertFalse(backend.needs_quantizer(f"{LINEAR}.qweight"))
        self.assertEqual(seen, [f"{LINEAR}.qweight"])

    def test_an_unquantized_checkpoint_never_consults_a_quantizer(self):
        backend = backend_for(None)
        self.assertFalse(backend.needs_quantizer(f"{LINEAR}.weight"))

    def test_materialize_reports_what_it_placed(self):
        """The caller has to know what to send back to meta once the module has run."""
        model, backend = self.build(None)
        weight = only_weight(backend, dense_shard())
        self.assertEqual(weight.materialize("cpu"), [f"{LINEAR}.weight"])


class TestPreparationHooks(unittest.TestCase):
    """The shard-level work a format does before any tensor can be placed."""

    def test_most_formats_leave_a_shard_alone(self):
        for config in (None, GPTQ_CONFIG, BNB_CONFIG):
            shard = dense_shard()
            with self.subTest(config=config):
                self.assertIs(backend_for(config).prepare_layer(shard), shard)

    def test_compressed_tensors_leaves_a_shard_with_nothing_packed_alone(self):
        """No packed payload means no decompression, and no need for the reader package either."""
        shard = dense_shard()
        backend = detect_backend(CT_CONFIG, caps=FakeCaps(), model=None, hf_quantizer=object(),
                                 compute_dtype=torch.bfloat16, device="cpu")
        self.assertEqual(set(backend.prepare_layer(shard)), set(shard))

    def test_a_plain_weight_under_a_packed_module_gets_its_parameter_back(self):
        """Checkpoints list residual Linears as quantization targets and then ship them in bf16."""
        model = nn.Module()
        linear = nn.Linear(4, 4, bias=False)
        del linear._parameters["weight"]
        linear.register_parameter("weight_packed", nn.Parameter(torch.zeros(4, 2,
                                                                            dtype=torch.uint8),
                                                                requires_grad=False))
        linear.quantization_scheme = object()
        model.add_module("block", linear)

        backend = detect_backend(CT_CONFIG, caps=FakeCaps(), model=model, hf_quantizer=object(),
                                 compute_dtype=torch.bfloat16, device="cpu")
        backend.prepare_layer({"block.weight": torch.zeros(4, 4, dtype=torch.bfloat16)})

        self.assertIn("weight", model.block._parameters)
        self.assertNotIn("weight_packed", model.block._parameters)
        self.assertFalse(hasattr(model.block, "quantization_scheme"))

    def test_a_missing_reader_package_names_itself(self):
        """There is no correct fallback for an undecodable payload, so it must say what to install."""
        backend = detect_backend(CT_CONFIG, caps=FakeCaps(), model=nn.Module(),
                                 hf_quantizer=object(), compute_dtype=torch.bfloat16, device="cpu")
        try:
            import compressed_tensors  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError) as raised:
                backend.prepare_layer(compressed_tensors_shard())
            self.assertIn("compressed-tensors", str(raised.exception))
        else:
            self.skipTest("compressed-tensors is installed, so the missing-reader path cannot run")

    def test_a_module_that_reads_packed_weights_is_left_packed(self):
        """CompressedLinear decompresses inside its own forward; expanding first would be wrong."""
        backend = backend_for(CT_CONFIG)
        # Without the reader package installed there is no such class, and saying "no" is the
        # honest answer: nothing here can consume a packed payload.
        self.assertIsInstance(backend.consumes_packed(nn.Linear(4, 4)), bool)
        self.assertFalse(backend.consumes_packed(nn.Linear(4, 4)))


class TestDecisionTable(unittest.TestCase):
    """What every known format would do here, reported before a long run rather than after."""

    def test_every_format_decides_and_explains_itself(self):
        for row in decision_table(caps=FakeCaps(), compute_dtype=torch.bfloat16):
            with self.subTest(fmt=row["format"]):
                self.assertIn(row["path"], ("packed", "dequant_to_scratch", "as_stored"))
                self.assertTrue(row["reason"].strip())
                self.assertTrue(row["capability"].strip())
                self.assertIsInstance(row["available"], bool)

    def test_the_table_turns_over_when_the_machine_does(self):
        bare = {row["format"]: row["path"]
                for row in decision_table(caps=FakeCaps(**NOTHING), compute_dtype=torch.bfloat16)}
        full = {row["format"]: row["path"]
                for row in decision_table(caps=FakeCaps(**EVERYTHING), compute_dtype=torch.bfloat16)}
        quantized = [fmt for fmt in bare if fmt != "dense"]
        self.assertTrue(quantized)
        for fmt in quantized:
            with self.subTest(fmt=fmt):
                self.assertEqual(bare[fmt], "dequant_to_scratch")
                self.assertEqual(full[fmt], "packed")
        self.assertEqual(bare["dense"], full["dense"], "an unquantized weight has no such decision")

    def test_the_format_announces_its_path_once_not_per_layer(self):
        C.reset_announcements()
        backend = backend_for(GPTQ_CONFIG)
        logger = logging.getLogger("rocketllm.hw.caps")
        with self.assertLogs(logger, level=logging.INFO) as captured:
            for _ in range(20):
                announce_backend(backend)
            logger.info("sentinel")
        C.reset_announcements()
        self.assertEqual(len(captured.output), 2, f"unexpected: {captured.output}")
        self.assertIn("dequant to scratch", captured.output[0])


class TestTensorSpec(unittest.TestCase):
    def test_a_spec_measures_the_same_bytes_as_the_tensor_it_describes(self):
        for dtype in (torch.bfloat16, torch.float32, torch.uint8, torch.int32):
            tensor = torch.zeros(7, 5, dtype=dtype)
            with self.subTest(dtype=dtype):
                spec = TensorSpec.of("t", tensor)
                self.assertEqual(spec.numel, tensor.numel())
                self.assertEqual(spec.nbytes, tensor.numel() * tensor.element_size())

    def test_a_weight_describes_itself_for_a_log_line(self):
        weight = only_weight(backend_for(GPTQ_CONFIG), gptq_shard())
        described = weight.describe()
        for key in ("name", "format", "tensors", "bits", "packed_bytes", "expanded_bytes",
                    "needs_scratch"):
            self.assertIn(key, described)
        self.assertIsInstance(weight, PackedWeight)
        self.assertIn("gptq", repr(weight))


class TestOnTheFlyQuantizationIsGone(unittest.TestCase):
    """RocketLLM imports pre-quantized checkpoints; it no longer makes them.

    The argument that used to request it is kept in the signature on purpose. Dropping it outright
    would raise TypeError, which tells a user their call is malformed rather than that the feature
    moved -- and the useful answer is where to get a checkpoint that is already quantized.
    """

    def test_asking_for_compression_raises_rather_than_being_ignored(self):
        """Silently ignoring it would stream 16-bit weights to someone who believes otherwise."""
        for requested in ("4bit", "8bit", "nf4"):
            with self.subTest(compression=requested):
                with self.assertRaises(ValueError):
                    reject_compression_argument(requested)

    def test_no_compression_requested_is_the_normal_path(self):
        self.assertIsNone(reject_compression_argument(None))

    def test_the_error_names_the_formats_that_do_work(self):
        """An error that only says no costs the user the search this message saves them."""
        with self.assertRaises(ValueError) as raised:
            reject_compression_argument("4bit")
        message = str(raised.exception)

        self.assertIn("4bit", message, "the message must quote what was actually asked for")
        for fmt in ("AWQ", "GPTQ", "compressed-tensors", "MXFP4", "bitsandbytes"):
            self.assertIn(fmt, message, f"{fmt} is supported but the message does not mention it")
        # The two things a user has to do next: find such a checkpoint, and install its reader.
        self.assertIn("rocketllm[quant]", message)
        self.assertIn("drop the compression= argument", message.lower())

    def test_the_engine_no_longer_carries_a_quantizing_code_path(self):
        """A leftover helper is an invitation to wire it back up."""
        import rocketllm.utils as utils

        for gone in ("compress_layer_state_dict", "uncompress_layer_state_dict",
                     "save_quant_state_to_dict"):
            with self.subTest(symbol=gone):
                self.assertFalse(hasattr(utils, gone))

    def test_nothing_in_the_engine_imports_bitsandbytes_to_load_a_model(self):
        """bitsandbytes is the reader for one format now, not a dependency of the engine.

        Reloading the modules with it blocked is what proves it, rather than trusting that this
        machine happens not to have it installed. ``rocketllm.base`` is deliberately not reloaded:
        it would rebind RocketModel to a second class object for the rest of the session, so what
        is checked there is that the import-time flag it used to keep is gone.
        """
        import importlib
        import sys

        import rocketllm.base as base

        self.assertFalse(hasattr(base, "bitsandbytes_installed"))

        blocked = {name: None for name in sys.modules if name.startswith("bitsandbytes")}
        saved = {name: sys.modules[name] for name in blocked}
        sys.modules.update(blocked)
        try:
            for module in ("rocketllm.utils", "rocketllm.quant.registry",
                           "rocketllm.quant.safetensors_quant"):
                with self.subTest(module=module):
                    self.assertIsNotNone(importlib.reload(importlib.import_module(module)))
        finally:
            sys.modules.update(saved)
            for name in blocked:
                if sys.modules.get(name) is None:
                    del sys.modules[name]


if __name__ == "__main__":
    unittest.main(verbosity=2)
