"""The pre-quantized formats RocketLLM reads out of safetensors shards.

Each backend here answers the same questions for a different on-disk layout: which checkpoint
tensors make up one logical weight, what that weight costs packed and expanded, and which device
capability decides whether it can be computed on as stored.

None of them quantize anything. Where transformers already knows how to reconstruct a parameter --
which is every format it ships a quantizer for -- the reconstruction is delegated to that quantizer
rather than reimplemented, because a second implementation of AWQ's packing order is a second thing
to get wrong. What lives here is the part transformers does not do for a streaming engine: grouping
a shard into logical weights, sizing them in packed bytes, and expanding the formats whose payload
no module can consume.

Every optional reader is imported where it is used, not at module import, so a machine without
compressed-tensors or bitsandbytes can still load every other format.
"""
import logging

import torch

from ..hw.caps import announce_once
from .registry import QuantBackend, quant_method_of, register_backend

log = logging.getLogger(__name__)


class HfQuantizerBackend(QuantBackend):
    """Formats transformers already knows how to reconstruct.

    ``preprocess_model`` has rebuilt the target modules to hold whatever parameters the scheme
    needs, and ``create_quantized_param`` knows how to fill them from a payload plus its companions.
    So placement is delegated -- the base class already does exactly that for any tensor the
    quantizer claims -- and what this class adds is the honest per-device answer about whether the
    result can be multiplied while still packed.

    This is also the landing place for a format this build has never heard of: if transformers
    wired up a quantizer for it, the delegation works without RocketLLM knowing the format's name.
    """

    FORMAT = "hf-quantizer"
    #: Fused kernels that can serve this format, best first. ``None`` means any 4-bit kernel will
    #: do, which is only true for a format this build does not recognise and cannot be specific
    #: about.
    FUSED_PROVIDERS = None

    @property
    def format(self):
        # Report what the checkpoint calls itself rather than which class happens to handle it;
        # a bug report saying "awq" is worth more than one saying "hf-quantizer".
        return quant_method_of(self.config) or self.FORMAT

    def fused_kernel(self):
        """The fused kernel that would serve this format here, if any.

        Not merely any fused kernel. Each format packs its values its own way, so only a kernel
        written for that packing can multiply it while it stays packed -- a bitsandbytes weight is
        not made computable by a Marlin kernel being installed. Reporting otherwise would promise a
        path the matmul cannot take.
        """
        if not self.FUSED_PROVIDERS:
            plan = self.caps.fused_4bit_plan()
            return plan.kernel if plan.fused else None
        providers = self.caps.fused_4bit_providers()
        return next((name for name in self.FUSED_PROVIDERS if providers.get(name)), None)

    def needs_scratch(self, weight):
        """Sub-16-bit weights need a fused kernel to be multiplied as stored.

        The kernel has to both exist and be runnable here, which is one query, not two guesses --
        an installed package built against another CUDA is worse than an absent one. Without it the
        payload is expanded into scratch and the matmul runs in the compute dtype: same answer,
        more bytes.
        """
        bits = self.bits(weight)
        if bits >= 16:
            return False
        if bits == 8 and weight.payload.is_floating_point:
            return not self.caps.supports_fp8
        return self.fused_kernel() is None

    def decision(self):
        kernel = self.fused_kernel()
        wanted = ", ".join(self.FUSED_PROVIDERS) if self.FUSED_PROVIDERS else "any 4-bit kernel"
        return {
            "format": self.format,
            "capability": f"a fused 4-bit kernel this format can use ({wanted}), on a backend that "
                          f"can run it",
            "available": kernel is not None,
            "path": "packed" if kernel else "dequant_to_scratch",
            "reason": (f"{kernel} is usable here, so weights stay packed through the matmul"
                       if kernel else
                       f"no usable fused kernel for this format ({wanted}), so each payload is "
                       f"expanded into a scratch buffer and the matmul runs in the compute dtype"),
        }


@register_backend
class GptqAwqBackend(HfQuantizerBackend):
    """GPTQ and AWQ: an int32-packed payload with group scales, zero points and an index.

    The two formats differ in the order they pack rows into those int32s, which matters only to the
    kernel that unpacks them -- and that kernel is the quantizer's, not ours. What they share is the
    layout this class cares about: one logical weight per Linear, spread over four tensors none of
    which is named ``weight``.
    """

    FORMAT = "gptq/awq"
    quant_methods = ("gptq", "awq", "auto-gptq", "autoawq", "gptq_marlin", "awq_marlin")
    #: Kernels that understand int32-packed 4-bit rows with group scales.
    FUSED_PROVIDERS = ("gptqmodel", "marlin_kernels", "exllamav2", "awq_ext", "torch_int4pack")

    #: The tensor that is the weight, and the ones that only describe it.
    PAYLOAD_LEAVES = ("qweight",)
    COMPANION_LEAVES = ("qzeros", "scales", "g_idx", "zeros", "scale")

    @classmethod
    def example_config(cls):
        return {"quant_method": "gptq", "bits": 4}

    def logical_name(self, tensor_name, known=()):
        module, _, leaf = tensor_name.rpartition(".")
        if module and leaf in self.PAYLOAD_LEAVES + self.COMPANION_LEAVES:
            return f"{module}.weight"
        return super().logical_name(tensor_name, known)

    def is_payload(self, tensor_name, logical_name):
        return tensor_name.rpartition(".")[2] in self.PAYLOAD_LEAVES or \
            super().is_payload(tensor_name, logical_name)


@register_backend
class CompressedTensorsBackend(HfQuantizerBackend):
    """compressed-tensors and MXFP4: a packed payload plus the scales that decode it.

    compressed-tensors ships no quantized compute kernels. Its own answer is to register a hook
    that decompresses the entire model before the first forward, which is exactly wrong here: it
    would materialise every expert of every layer at once -- tens of gigabytes for one layer of a
    large MoE -- to run a token that touches a fraction of them. So the base model removes that hook
    and this backend expands each shard as it arrives instead, on the device, after transferring the
    *packed* bytes. For MXFP4 that is four times less data across the link than sending an already
    expanded weight.
    """

    FORMAT = "compressed-tensors"
    quant_methods = ("compressed-tensors", "compressed_tensors", "mxfp4")

    #: MXFP4 stores its payload under ``_blocks``; compressed-tensors under ``_packed``.
    PAYLOAD_SUFFIXES = ("_packed", "_blocks")

    @classmethod
    def example_config(cls):
        return {"quant_method": "compressed-tensors",
                "config_groups": {"group_0": {"weights": {"num_bits": 4}}}}

    def needs_scratch(self, weight):
        """For a packed payload here, native 4-bit arithmetic is the only alternative to expanding.

        There is no kernel package to fall back on -- compressed-tensors does not ship one -- so
        unlike GPTQ or AWQ this format cannot be rescued by installing something. Either the device
        multiplies the stored type natively or the weight is expanded first.

        This answers for the *device*. A module still gets expanded regardless when its own forward
        reads a plain ``weight``, which is what compressed-tensors patches onto every quantized
        Linear it did not build as a CompressedLinear. See :meth:`consumes_packed`.
        """
        bits = self.bits(weight)
        if bits >= 16:
            return False
        if bits == 8 and weight.payload.is_floating_point:
            return not self.caps.supports_fp8
        return not self.caps.supports_fp4

    def decision(self):
        native = self.caps.supports_fp4
        return {
            "format": self.format,
            "capability": "native 4-bit arithmetic (compressed-tensors ships no fused kernels)",
            "available": native,
            "path": "packed" if native else "dequant_to_scratch",
            "reason": ("the device multiplies the stored 4-bit type natively, so the payload can "
                       "stay packed for a module whose forward reads it"
                       if native else
                       "no native 4-bit arithmetic here, so each packed payload is expanded into a "
                       "scratch buffer on the device just before its layer runs and freed after"),
        }

    # -- expansion -------------------------------------------------------------------------------

    def consumes_packed(self, module):
        """Whether this module's own forward reads the packed payload.

        compressed-tensors builds a CompressedLinear when the checkpoint is loaded to run
        compressed; its forward decompresses internally, so handing it an expanded weight would be
        both wasteful and wrong. Every other quantized Linear gets a patched forward that reads a
        plain ``self.weight``, and for those the expansion is not optional whatever the device can
        do.
        """
        try:
            from compressed_tensors.linear.compressed_linear import CompressedLinear
        except ImportError:
            return False
        return isinstance(module, CompressedLinear)

    def prepare_layer(self, state_dict):
        """Expand packed payloads into the plain weights the modules expect."""
        state_dict = self._restore_plain_weight_modules(state_dict)
        return self._decompress(state_dict)

    def _restore_plain_weight_modules(self, state_dict):
        """Undo the packed-parameter layout where the checkpoint still ships a plain ``weight``.

        Some checkpoints list residual and router Linears under the quantization target list but
        store them at full precision anyway. After ``preprocess_model`` those modules expose
        ``weight_packed`` and reject the real ``weight`` tensor, so a meta ``weight`` Parameter is
        put back and the shard loads.
        """
        if self.model is None:
            return state_dict

        plain = {k[: -len(".weight")] for k in state_dict if k.endswith(".weight")}
        packed = {k[: -len(".weight_packed")] for k in state_dict if k.endswith(".weight_packed")}
        for prefix in plain - packed:
            try:
                module = self.model.get_submodule(prefix)
            except AttributeError:
                continue
            names = list(module._parameters.keys())
            if "weight" in names or "weight_packed" not in names:
                continue
            weight = state_dict[f"{prefix}.weight"]
            for name in names:
                if name == "weight" or name.startswith("weight_"):
                    module._parameters.pop(name, None)
            module.register_parameter(
                "weight",
                torch.nn.Parameter(torch.empty(weight.shape, device="meta", dtype=weight.dtype),
                                   requires_grad=False),
            )
            for attr in ("quantization_scheme", "quantization_status", "quantization_format"):
                if hasattr(module, attr):
                    delattr(module, attr)
        return state_dict

    def _decompress(self, state_dict):
        """Expand this shard's packed payloads, on the device, after the packed bytes have moved."""
        if self.hf_quantizer is None or self.model is None:
            return state_dict

        packed_prefixes = {k[: -len(".weight_packed")]
                           for k in state_dict if k.endswith(".weight_packed")}
        if not packed_prefixes:
            return state_dict

        try:
            from compressed_tensors.compressors.base import BaseCompressor
        except ImportError as exc:
            # There is no slower-but-correct path here: nothing else can decode this payload. So
            # this is the one place in the package that refuses, and it says what to install.
            raise ImportError(
                "this checkpoint stores packed weights that only compressed-tensors can decode, "
                "and it is not installed. Install it with `pip install rocketllm[quant]` or "
                f"`pip install compressed-tensors` and load the model again. ({exc})") from exc

        out = dict(state_dict)
        for prefix in packed_prefixes:
            try:
                module = self.model.get_submodule(prefix)
            except AttributeError:
                continue
            if self.consumes_packed(module):
                continue
            scheme = getattr(module, "quantization_scheme", None)
            if scheme is None or getattr(scheme, "format", None) is None:
                continue

            local = {k[len(prefix) + 1:]: v.to(self.device)
                     for k, v in state_dict.items() if k.startswith(prefix + ".")}
            # scheme.format is an enum on some compressed-tensors versions and a plain str on others.
            fmt = getattr(scheme.format, "value", scheme.format)
            compressor = BaseCompressor.get_value_from_registry(fmt)
            decompressed = compressor.decompress(local, scheme)

            for k in local:
                out.pop(f"{prefix}.{k}", None)
            if "weight" in decompressed:
                # Once expanded the module reads only `weight`; the scales that came back merely
                # describe how the checkpoint stored it, and keeping them would waste VRAM.
                out[f"{prefix}.weight"] = decompressed["weight"]
                self._expose_plain_weight(module, decompressed["weight"])
            else:
                for k, v in decompressed.items():
                    out[f"{prefix}.{k}"] = v
        return out

    @staticmethod
    def _expose_plain_weight(module, weight):
        """Swap a module's packed parameters for the plain ``weight`` its forward reads.

        Marking the module COMPRESSED tells that forward the weight is already on the quantization
        grid, so it skips a fake-quantize that would only reproduce what was just decompressed.
        """
        existing = module._parameters.get("weight")
        if existing is None or existing.shape != weight.shape:
            for name in [n for n in list(module._parameters) if n.startswith("weight")]:
                module._parameters.pop(name, None)
            module.register_parameter(
                "weight",
                torch.nn.Parameter(torch.empty(weight.shape, device="meta", dtype=weight.dtype),
                                   requires_grad=False),
            )
        try:
            from compressed_tensors.quantization import QuantizationStatus
        except ImportError:
            return
        module.quantization_status = QuantizationStatus.COMPRESSED


@register_backend
class BitsAndBytesBackend(HfQuantizerBackend):
    """bitsandbytes-prequantized weights: a payload plus the quant state that decodes it.

    The quant state is a handful of small tensors stored under the weight's own name, which is why
    grouping here is structural: a tensor whose parent name is itself a tensor in the same shard is
    part of that weight, whatever the two are called. RocketLLM's own shards write the same
    relationship with a ``.4bit.``/``.8bit.`` marker, and both spellings group the same way.

    bitsandbytes is the kernel as well as the format, so whether these weights compute packed is the
    same question as whether bitsandbytes imports and runs here.
    """

    FORMAT = "bitsandbytes"
    quant_methods = ("bitsandbytes", "bitsandbytes_4bit", "bitsandbytes_8bit", "bnb_4bit",
                     "bnb_8bit")
    #: Only its own kernel can read its own packing, so this is the one format with no alternative.
    FUSED_PROVIDERS = ("bitsandbytes",)

    @classmethod
    def example_config(cls):
        return {"quant_method": "bitsandbytes_4bit", "bits": 4}

    def declared_bits(self):
        """The method name carries the width when the config does not spell it out."""
        declared = super().declared_bits()
        if declared:
            return declared
        method = quant_method_of(self.config) or ""
        if "4bit" in method:
            return 4
        if "8bit" in method:
            return 8
        return None

    def logical_name(self, tensor_name, known=()):
        base = super().logical_name(tensor_name, known)
        if base != tensor_name:
            return base
        # A quant-state tensor is stored under the weight it belongs to, so the weight's own name
        # is a prefix of it. Checking against the shard rather than against a list of quant-state
        # names keeps this working when bitsandbytes adds a field.
        parent = tensor_name.rpartition(".")[0]
        while parent:
            if parent in known:
                return parent
            parent = parent.rpartition(".")[0]
        return base

    def is_consumed(self, tensor_name, logical_name):
        """The quant state is read out of the shard, not placed: it extends the weight's name."""
        if super().is_consumed(tensor_name, logical_name):
            return True
        return tensor_name != logical_name and tensor_name.startswith(logical_name + ".")


def announce_backend(backend):
    """State once what this checkpoint's format will do on this machine.

    Loading a very large model is a long wait, and "it dequantized every layer because this build
    has no fused kernel" is the difference between a slow run and a broken machine. The user should
    learn it at load, not infer it from the clock.
    """
    decision = backend.decision()
    announce_once(
        f"quant-{backend.format}-{decision['path']}",
        f"{backend.format} checkpoint: {decision['path'].replace('_', ' ')} -- "
        f"{decision['reason']}",
        logging.INFO)
