"""A KIVI-style int4 KV cache, and the choice of whether to use one.

The KV cache is the other thing competing for device memory, and unlike the weights it grows with
every token. In a streaming engine that makes it directly adversarial: every byte the cache takes is
a byte the weight cache cannot pin, so a long context does not merely cost memory, it costs storage
reads on every subsequent token. Quantizing it buys that memory back.

This is INDEPENDENT of the weight format, and nothing here may consult it. A model whose weights are
AWQ, GPTQ, MXFP4 or plain bf16 makes exactly the same KV decision, because the two answer different
questions: the weight format is a property of the checkpoint someone else produced, and this is a
property of how much room the running machine has for a context.

The recipe
----------
The asymmetry between K and V is the whole point, and it is not arbitrary.

**K is quantized per channel.** Key projections have persistent per-channel outliers: a handful of
head_dim positions carry values an order of magnitude larger than the rest, in the same positions for
every token. Quantizing a token's key across its channels puts those outliers in the same group as
ordinary values, the group's range stretches to cover them, and every ordinary value in it loses
most of its resolution. Giving each channel its own scale confines the damage to the channels that
actually have the outliers. Naive per-token int4 on K is visibly worse -- it is the single mistake
that makes a 4-bit KV cache look unusable, and it is why this file does not simply quantize both
tensors the same way.

**V is quantized per token.** Value projections have no comparable channel structure, so the extra
machinery buys nothing, and per-token grouping keeps the scales next to the tokens they describe.

Concretely, with K shaped ``[batch, heads, tokens, channels]``: K groups along the TOKEN axis, so a
scale belongs to one channel and covers `group_size` consecutive tokens. V groups along the CHANNEL
axis, so a scale belongs to one token and covers `group_size` consecutive channels.

The most recent tokens stay in an fp16 residual window. They are the ones attention weights most
sharply, they are too few to be worth compressing, and quantizing K per channel needs a whole group
of tokens before it can compute a scale at all -- so the window is not only a quality decision, it is
what makes the layout possible.

What this does not do
---------------------
It does not make generation faster. Attention needs the whole history in the compute dtype, so every
step dequantizes every chunk, and that is work a plain fp16 cache does not do. The trade is memory
for time, deliberately: on a machine where the weights already do not fit, the memory is worth more.
Getting the time back needs an attention kernel that reads the packed form directly, which is a
different piece of work.
"""
import dataclasses
import logging

import torch
from transformers.cache_utils import Cache

from ..hw import caps

log = logging.getLogger(__name__)

#: Settings a KV cache can be asked for. Strings rather than an enum because they arrive from a
#: constructor argument and an environment override, and both are text.
KV_AUTO = "auto"
KV_FP16 = "fp16"
KV_INT4 = "int4"
#: Delegating modes: transformers' own QuantizedCache on one of its backends, kept as a reference to
#: check this implementation against rather than as the default.
KV_HQQ = "hqq"
KV_QUANTO = "quanto"

KV_CHOICES = (KV_AUTO, KV_FP16, KV_INT4, KV_HQQ, KV_QUANTO)
#: Delegating mode -> (module to import, package to install). The two differ for quanto.
_DELEGATED = {KV_HQQ: ("hqq", "hqq"), KV_QUANTO: ("optimum.quanto", "optimum-quanto")}

#: int4, so sixteen levels. Kept as a name because the packing below assumes two codes per byte.
_LEVELS = 15
_BITS = 4


@dataclasses.dataclass(frozen=True)
class KVCacheConfig:
    """How a quantized KV cache is laid out.

    Defaults follow the recipe; every one of them is overridable, and the engine supplies
    `residual_length` and `group_size` from the hardware profile rather than from here.
    """

    group_size: int = 64
    residual_length: int = 128
    compute_dtype: object = torch.float16
    #: Axis each tensor groups along, as a dimension of ``[batch, heads, tokens, channels]``.
    #: K groups along tokens so a scale belongs to a channel; V groups along channels so a scale
    #: belongs to a token. Swapping these is the bug this file exists to avoid.
    key_axis: int = 2
    value_axis: int = 3

    def __post_init__(self):
        if self.group_size < 2:
            raise ValueError(f"group_size must be at least 2, not {self.group_size}")
        if self.residual_length < self.group_size:
            # The window has to be able to hold a whole group, or no group can ever be completed
            # and nothing is ever quantized.
            raise ValueError(f"residual_length {self.residual_length} is below group_size "
                             f"{self.group_size}; the window could never complete a group")


# -- the quantizer -----------------------------------------------------------------------------

def _pad_to_group(tensor, axis, group):
    """Extend `axis` to a multiple of `group` by repeating its last element.

    Repetition rather than zero padding, because the padding is inside the group whose scale is
    about to be computed: a zero would widen the range of a group that does not contain one and
    lower the resolution of every real value in it. A duplicate of a value already there cannot
    change the group's min or max, so the scale is exactly what it would have been.
    """
    length = tensor.shape[axis]
    remainder = (-length) % group
    if not remainder:
        return tensor, 0
    tail = tensor.narrow(axis, length - 1, 1)
    filler = tail.expand(*[-1 if d != axis else remainder for d in range(tensor.dim())])
    return torch.cat([tensor, filler], dim=axis), remainder


def quantize(tensor, axis, group_size):
    """Group `tensor` along `axis` and encode each group as asymmetric int4.

    Returns everything needed to invert it. Codes are packed two per byte -- storing one code per
    byte would cost the same as int8 and give back half the saving this exists for.
    """
    tensor = tensor.contiguous()
    original_length = tensor.shape[axis]
    group = min(group_size, original_length) if original_length else group_size
    padded, padding = _pad_to_group(tensor, axis, group)

    # Move the grouped axis last, then split it into (groups, group) so the reduction is over one
    # trailing dimension whatever axis the caller asked for.
    moved = padded.movedim(axis, -1)
    shape = moved.shape
    grouped = moved.reshape(*shape[:-1], shape[-1] // group, group)

    lo = grouped.amin(dim=-1, keepdim=True)
    hi = grouped.amax(dim=-1, keepdim=True)
    scale = (hi - lo) / _LEVELS
    # A group whose values are all equal has no range. Any non-zero scale inverts it exactly, since
    # every code comes out zero and dequantizing gives back `lo`; leaving it at zero would divide.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))

    codes = torch.clamp(torch.round((grouped.float() - lo.float()) / scale.float()), 0, _LEVELS)
    packed = _pack(codes.to(torch.uint8).reshape(*shape[:-1], shape[-1]))

    return QuantizedTensor(
        packed=packed, scale=scale.squeeze(-1).to(torch.float16),
        zero=lo.squeeze(-1).to(torch.float16), axis=axis, group=group,
        length=original_length, padding=padding, shape=tuple(tensor.shape),
        dtype=tensor.dtype)


@dataclasses.dataclass
class QuantizedTensor:
    """One quantized block, and everything needed to put it back."""

    packed: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    axis: int
    group: int
    length: int
    padding: int
    shape: tuple
    dtype: object

    @property
    def nbytes(self):
        """What this block actually occupies, codes and scales together.

        The scales are not a rounding error: at group 64 they are a fp16 pair per 64 values, which
        is another half bit each. Reporting only the codes would overstate the saving by a tenth.
        """
        return (self.packed.numel() * self.packed.element_size()
                + self.scale.numel() * self.scale.element_size()
                + self.zero.numel() * self.zero.element_size())

    def dequantize(self):
        codes = _unpack(self.packed)
        shape = codes.shape
        grouped = codes.reshape(*shape[:-1], shape[-1] // self.group, self.group).to(torch.float32)
        restored = grouped * self.scale.unsqueeze(-1).float() + self.zero.unsqueeze(-1).float()
        flat = restored.reshape(*shape[:-1], shape[-1])
        if self.padding:
            flat = flat.narrow(-1, 0, flat.shape[-1] - self.padding)
        return flat.movedim(-1, self.axis).to(self.dtype)


def _pack(codes):
    """Two 4-bit codes per byte, along the last dimension."""
    if codes.shape[-1] % 2:
        codes = torch.cat([codes, torch.zeros_like(codes.narrow(-1, 0, 1))], dim=-1)
    low = codes[..., 0::2]
    high = codes[..., 1::2]
    return (low | (high << _BITS)).contiguous()


def _unpack(packed):
    low = packed & _LEVELS
    high = (packed >> _BITS) & _LEVELS
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


# -- the cache ---------------------------------------------------------------------------------

class QuantizedKVCache(Cache):
    """int4 KV cache with an fp16 residual window, in the shape transformers expects.

    Quantized blocks are append-only. Once a group of tokens has been encoded it is never touched
    again -- which is both faster than re-encoding the whole history whenever the window fills, and
    more accurate, because re-quantizing an already-dequantized block compounds the error of every
    pass it has been through.
    """

    is_compileable = False

    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else KVCacheConfig()
        #: Per layer: the list of quantized (key, value) blocks, oldest first.
        self._blocks = []
        #: Per layer: the fp16 residual window, newest tokens.
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0
        self.flushes = 0

    # -- transformers Cache surface --------------------------------------------------------------

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        self._ensure(layer_idx, key_states)

        residual_keys = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
        residual_values = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        residual_keys, residual_values = self._flush(layer_idx, residual_keys, residual_values)
        self.key_cache[layer_idx] = residual_keys
        self.value_cache[layer_idx] = residual_values

        keys, values = self._blocks[layer_idx]
        if not keys:
            return residual_keys, residual_values
        # Attention needs the whole history in the compute dtype, so the blocks are expanded on
        # every step. This is the time this trade costs; see the module docstring.
        return (torch.cat([block.dequantize() for block in keys] + [residual_keys], dim=-2),
                torch.cat([block.dequantize() for block in values] + [residual_values], dim=-2))

    def _ensure(self, layer_idx, like):
        while len(self.key_cache) <= layer_idx:
            empty = torch.zeros(like.shape[:-2] + (0, like.shape[-1]),
                                dtype=like.dtype, device=like.device)
            self.key_cache.append(empty)
            self.value_cache.append(empty.clone())
            self._blocks.append(([], []))

    def _flush(self, layer_idx, keys, values):
        """Move whole groups of the oldest residual tokens into quantized blocks.

        Only whole groups, and only the ones past the window. K's scales belong to a channel and
        cover `group_size` consecutive tokens, so a partial group cannot be encoded without either
        inventing tokens or giving that group a scale computed from fewer of them -- and the second
        makes the boundary between residual and quantized behave differently from everywhere else,
        which is exactly where these implementations go wrong.
        """
        excess = keys.shape[-2] - self.config.residual_length
        if excess <= 0:
            return keys, values
        take = (excess // self.config.group_size) * self.config.group_size
        if take <= 0:
            return keys, values

        block_keys, block_values = self._blocks[layer_idx]
        block_keys.append(quantize(keys.narrow(-2, 0, take), self.config.key_axis,
                                   self.config.group_size))
        block_values.append(quantize(values.narrow(-2, 0, take), self.config.value_axis,
                                     self.config.group_size))
        self.flushes += 1
        remaining = keys.shape[-2] - take
        return (keys.narrow(-2, take, remaining).contiguous(),
                values.narrow(-2, take, remaining).contiguous())

    def get_seq_length(self, layer_idx=0):
        if layer_idx >= len(self.key_cache):
            return 0
        quantized = sum(block.length for block in self._blocks[layer_idx][0])
        return quantized + self.key_cache[layer_idx].shape[-2]

    def get_max_cache_shape(self):
        return None

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx):
        for layer_idx in range(len(self.key_cache)):
            for name in ("key_cache", "value_cache"):
                tensor = getattr(self, name)[layer_idx]
                if tensor.numel():
                    getattr(self, name)[layer_idx] = tensor.index_select(
                        0, beam_idx.to(tensor.device))
            for blocks in self._blocks[layer_idx]:
                for block in blocks:
                    block.packed = block.packed.index_select(0, beam_idx.to(block.packed.device))
                    block.scale = block.scale.index_select(0, beam_idx.to(block.scale.device))
                    block.zero = block.zero.index_select(0, beam_idx.to(block.zero.device))

    @property
    def seen_tokens(self):
        return self._seen_tokens

    def __len__(self):
        return len(self.key_cache)

    # -- what the engine asks ---------------------------------------------------------------------

    def nbytes(self):
        """Device bytes this cache is holding, quantized blocks and residual window together."""
        total = 0
        for layer_idx in range(len(self.key_cache)):
            for blocks in self._blocks[layer_idx]:
                total += sum(block.nbytes for block in blocks)
            for name in ("key_cache", "value_cache"):
                tensor = getattr(self, name)[layer_idx]
                total += tensor.numel() * tensor.element_size()
        return total

    def report(self):
        layers = len(self.key_cache)
        residual = sum(self.key_cache[i].shape[-2] for i in range(layers)) // max(1, layers)
        quantized = sum(block.length for block in self._blocks[0][0]) if layers else 0
        return {
            "kind": "int4",
            "layers": layers,
            "tokens": self.get_seq_length(0),
            "quantized_tokens": quantized,
            "residual_tokens": residual,
            "group_size": self.config.group_size,
            "residual_length": self.config.residual_length,
            "flushes": self.flushes,
            "bytes": self.nbytes(),
        }


# -- choosing one ------------------------------------------------------------------------------

def resolve_kv_cache(setting, weight_bytes=None, device_bytes=None, headroom=0.0, profile=None):
    """Turn a `kv_cache=` setting into a concrete one, and say why.

    Returns ``(choice, reason)``. Only ``"auto"`` is decided here; everything else is the user's
    instruction and is passed through, because an explicit setting that quietly became something
    else would be worse than either answer.

    The auto rule is about whether memory is the binding constraint, and that is a question about
    the model and the machine together -- the profile alone cannot answer it, since the same card is
    roomy for one checkpoint and hopeless for the next. So: if the weights fit resident with the
    configured headroom left over, nothing is being traded away by keeping the context exact, and
    fp16 is the better cache. If they do not, the device is already the bottleneck, every byte the
    context takes is a byte of weights that has to be re-read on the next token, and int4 buys back
    roughly three and a half times the context per byte.
    """
    if setting not in KV_CHOICES:
        raise ValueError(f"kv_cache must be one of {', '.join(KV_CHOICES)}, not {setting!r}")
    if setting != KV_AUTO:
        return setting, "requested explicitly"

    if not weight_bytes or not device_bytes:
        # Nothing measured to decide from. fp16 is the honest default: it is what the model would
        # do untouched, and choosing lossy compression on no evidence is not a default anyone asked
        # for.
        return KV_FP16, ("no measured weight or device size to decide from, so the context is kept "
                         "exact")

    needed = weight_bytes * (1.0 + max(0.0, headroom))
    if needed <= device_bytes:
        return KV_FP16, (f"the weights fit resident ({weight_bytes / 1024 ** 3:.1f}GB with "
                         f"{headroom:.0%} headroom against a {device_bytes / 1024 ** 3:.1f}GB "
                         f"budget), so memory is not the binding constraint and the context is kept "
                         f"exact")
    return KV_INT4, (f"the weights do not fit resident ({weight_bytes / 1024 ** 3:.1f}GB against a "
                     f"{device_bytes / 1024 ** 3:.1f}GB budget), so every byte the context takes is "
                     f"a byte of weights re-read next token")


def build_kv_cache(choice, config=None, device=None, compute_dtype=None):
    """Build the cache a resolved choice names, or ``None`` for fp16.

    ``None`` means "let transformers do what it always does", which is a DynamicCache in the compute
    dtype. Returning None rather than a passthrough wrapper matters: it keeps the unquantized path
    byte-for-byte the stock one, so a correctness gate run at fp16 is testing transformers' cache and
    not ours.
    """
    if choice == KV_FP16:
        return None
    if choice == KV_INT4:
        return QuantizedKVCache(config)
    return _delegated_cache(choice, config, device, compute_dtype)


def _delegated_cache(choice, config, device, compute_dtype):
    """transformers' own QuantizedCache, as a reference to check this implementation against.

    Kept because "our int4 output looks plausible" is not a measurement, and running the same prompt
    through a separately written 4-bit KV cache is. Note that it is a *reference*, not an oracle:
    transformers quantizes K and V the same way, by its own documentation "in contrast to what was
    described in the paper", so it does not implement the per-channel/per-token split this module
    exists for. Where the two disagree, that difference is one of the candidate explanations, and
    the fp16 run is the thing both are measured against.

    Both backends are optional packages and neither is needed to load a model, so a missing one
    degrades to the cache the engine would otherwise have used and names what to install.
    """
    from transformers.cache_utils import (HQQQuantizedCache, QuantizedCacheConfig,
                                          QuantoQuantizedCache)

    config = config or KVCacheConfig()
    module, package = _DELEGATED[choice]
    try:
        __import__(module)
    except ImportError:
        caps.announce_once(
            f"kv-backend-{choice}",
            f"kv_cache={choice!r} needs the {package} package, which is not installed. Falling back "
            f"to RocketLLM's own int4 KV cache, which is the default anyway; install {package} only "
            f"to have the reference implementation to compare against.",
            logging.INFO)
        return QuantizedKVCache(config)

    backend_config = QuantizedCacheConfig(
        backend=choice, nbits=4, q_group_size=config.group_size,
        residual_length=config.residual_length,
        compute_dtype=compute_dtype or config.compute_dtype, device=str(device or "cpu"))
    return (HQQQuantizedCache(backend_config) if choice == KV_HQQ
            else QuantoQuantizedCache(backend_config))
