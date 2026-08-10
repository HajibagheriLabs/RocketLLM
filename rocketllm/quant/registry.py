"""The format-agnostic view of a pre-quantized checkpoint.

RocketLLM imports pre-quantized checkpoints; it never quantizes a model itself. What differs between
AWQ, GPTQ, compressed-tensors, MXFP4 and a bitsandbytes-prequantized file is how one logical weight
is spread across checkpoint tensors, and what has to happen to it before a matmul can read it.
Everything above this package wants the same answers regardless of which: the tensors a weight
spans, what it costs packed and expanded, and what to put on the device.

:class:`PackedWeight` is that answer, :class:`QuantBackend` is the per-format knowledge behind it,
and :func:`detect_backend` picks a backend from the checkpoint's own ``quantization_config`` -- from
what the file declares about itself, never from a model or architecture name.

One decision deliberately does not live in the format. Whether a weight stays packed through the
matmul is a property of the *machine*: the same AWQ checkpoint computes from the packed form on a
card with a fused kernel and has to be expanded into scratch on one without. So ``needs_scratch``
asks rocketllm.hw.caps every time, and two machines reading the same file will disagree about it,
correctly. What the format decides is *which* capability gets asked; the device decides the answer.
"""
import logging
from collections import OrderedDict

import torch
from accelerate.utils.modeling import set_module_tensor_to_device

from ..hw.caps import get_caps

log = logging.getLogger(__name__)

#: Suffixes of the companion tensors that pre-quantized checkpoints ship next to a payload: fp8
#: block scales, compressed-tensors/MXFP4 packed payloads and their scales and shapes, GPTQ indices.
#: Longest first, because stripping one to recover the logical name has to prefer ``_scale_inv``
#: over the ``_scale`` it ends with.
COMPANION_SUFFIXES = ("_scale_inv", "_zero_point", "_packed", "_scale", "_shape", "_g_idx")

#: Markers RocketLLM's own bitsandbytes shards use to attach a quant state to its weight.
BNB_MARKERS = (".4bit.", ".8bit.")

#: Keys a checkpoint may declare its weight width under. Different formats spell it differently and
#: compressed-tensors buries it under per-group schemes, so the search is by key, at any depth.
BIT_WIDTH_KEYS = ("bits", "w_bit", "num_bits", "weight_bits")


def _itemsize(dtype):
    size = getattr(dtype, "itemsize", None)
    if size is not None:
        return int(size)
    # torch < 2.1 has no dtype.itemsize; a zero-dim tensor is the portable way to ask.
    return int(torch.empty((), dtype=dtype).element_size())


class TensorSpec:
    """One checkpoint tensor, described without holding its data.

    The cache has to size a layer before deciding whether to keep it, and a safetensors header
    answers shape and dtype without reading a byte. So everything about size is expressed over
    specs, and only materialisation needs the tensors themselves.
    """

    __slots__ = ("name", "shape", "dtype")

    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = dtype

    @classmethod
    def of(cls, name, tensor):
        return cls(name, tuple(tensor.shape), tensor.dtype)

    @property
    def numel(self):
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def itemsize(self):
        return _itemsize(self.dtype)

    @property
    def nbytes(self):
        return self.numel * self.itemsize

    @property
    def is_floating_point(self):
        return self.dtype.is_floating_point

    def __repr__(self):
        return f"<TensorSpec {self.name} {tuple(self.shape)} {self.dtype}>"

    def __eq__(self, other):
        return (isinstance(other, TensorSpec) and other.name == self.name
                and other.shape == self.shape and other.dtype == self.dtype)

    def __hash__(self):
        return hash((self.name, self.shape, self.dtype))


class PackedWeight:
    """One logical weight, as the checkpoint stores it.

    A weight is rarely one tensor. GPTQ spreads it over ``qweight``/``qzeros``/``scales``/``g_idx``,
    compressed-tensors over ``weight_packed``/``weight_scale``/``weight_shape``, an fp8 checkpoint
    over ``weight`` plus ``weight_scale_inv``, bitsandbytes over a payload plus its quant state.
    Callers above this layer should not have to know which, so they get one object per logical
    weight that can answer for the whole group.

    Sizes are always in *packed* bytes -- what actually crosses the link and sits in the cache.
    ``expanded_bytes`` is what the same weight would cost once dequantized, which is what a scratch
    buffer has to be big enough for, not what the cache holds.
    """

    __slots__ = ("name", "specs", "backend", "_values", "_shard")

    def __init__(self, name, specs, backend, values=None, shard=None):
        self.name = name
        #: Payload first, then companions. The payload is the tensor the weight is really made of.
        self.specs = tuple(specs)
        self.backend = backend
        self._values = values
        # The shard this weight came out of. Quantizers that reconstruct a parameter read companion
        # tensors straight out of the state dict they were handed, and some look outside this
        # weight's own group, so the whole shard travels with it.
        self._shard = shard

    # -- what it is made of ----------------------------------------------------------------------

    @property
    def format(self):
        return self.backend.format

    @property
    def tensor_names(self):
        return tuple(spec.name for spec in self.specs)

    @property
    def payload(self):
        return self.specs[0]

    @property
    def companions(self):
        return self.specs[1:]

    @property
    def bits(self):
        """Stored width of one logical value, as the checkpoint declares it."""
        return self.backend.bits(self)

    @property
    def logical_shape(self):
        """Shape of the weight the module ends up with, or ``None`` if it cannot be recovered."""
        return self.backend.logical_shape(self)

    # -- size accounting -------------------------------------------------------------------------

    @property
    def packed_bytes(self):
        """Bytes this weight occupies as stored, companions included."""
        return sum(spec.nbytes for spec in self.specs)

    @property
    def expanded_bytes(self):
        """Bytes the same weight would occupy dequantized into the compute dtype.

        The companions are not counted: once a weight is expanded, the scales and zero points have
        done their job and describing how the checkpoint stored it is not worth VRAM.
        """
        return self.backend.expanded_bytes(self)

    @property
    def scratch_bytes(self):
        """What a scratch buffer must hold for this weight -- zero when it computes packed."""
        return self.expanded_bytes if self.needs_scratch else 0

    # -- the per-device decision -----------------------------------------------------------------

    @property
    def needs_scratch(self):
        """Whether this weight must be expanded before the running device can compute on it.

        Asked of the machine, not of the file. The format chooses which capability is relevant --
        a fused kernel for GPTQ, native fp4 for MXFP4 -- and rocketllm.hw.caps answers it.
        """
        return self.backend.needs_scratch(self)

    # -- materialisation -------------------------------------------------------------------------

    @property
    def has_values(self):
        return self._values is not None

    def value_of(self, tensor_name):
        if self._values is None:
            raise ValueError(
                f"{self.name} was planned from checkpoint metadata alone, so it has no tensor data "
                f"to place; build it from a state dict to materialise it")
        return self._values[tensor_name]

    def place(self, tensor_name, device, shard=None):
        """Put one of this weight's tensors where the module expects it."""
        return self.backend.place(tensor_name, self.value_of(tensor_name), device,
                                  shard if shard is not None else self._shard or self._values)

    @property
    def placed_names(self):
        """The tensors that become parameters of their own.

        Not all of them do. A quant state the quantizer reads back out of the shard describes the
        payload rather than standing beside it, and the module has nowhere to put it -- so it is
        spanned and counted, but never placed.
        """
        return [spec.name for spec in self.specs
                if not self.backend.is_consumed(spec.name, self.name)]

    def materialize(self, device, shard=None):
        """Put this weight on `device` in the form the module actually reads.

        Returns the parameter names that were placed, which is what the caller needs to send back
        to meta when the module has run.
        """
        placed = []
        for tensor_name in self.placed_names:
            self.place(tensor_name, device, shard)
            placed.append(tensor_name)
        return placed

    def placements(self):
        """The tensors that are ordinary parameter placements, with the dtype each lands in.

        Yields ``(name, tensor, dtype)``. Anything the quantizer reconstructs itself is absent --
        there is no plain buffer to pack for those, so they cannot be coalesced with the rest.
        """
        out = []
        for tensor_name in self.placed_names:
            if self.backend.needs_quantizer(tensor_name):
                continue
            value = self.value_of(tensor_name)
            out.append((tensor_name, value, self.backend.target_dtype(tensor_name, value)))
        return out

    def quantizer_names(self):
        """The tensors the quantizer reconstructs rather than the loader placing them."""
        return [name for name in self.placed_names if self.backend.needs_quantizer(name)]

    # -- reporting -------------------------------------------------------------------------------

    def describe(self):
        return {
            "name": self.name,
            "format": self.format,
            "tensors": list(self.tensor_names),
            "bits": self.bits,
            "packed_bytes": self.packed_bytes,
            "expanded_bytes": self.expanded_bytes,
            "needs_scratch": self.needs_scratch,
            "logical_shape": self.logical_shape,
        }

    def __repr__(self):
        return (f"<PackedWeight {self.name} {self.format} {len(self.specs)} tensors "
                f"{self.packed_bytes}B packed>")


class QuantBackend:
    """How one storage format becomes weights on a device.

    Subclasses answer three kinds of question: how the format lays a logical weight out across
    checkpoint tensors, how big it is packed and expanded, and which device capability decides
    whether it can be computed on as stored. Placement itself is shared -- it is the same
    ``set_module_tensor_to_device`` call for every format, or the quantizer's own reconstruction --
    so only formats that genuinely differ override it.

    This base class is also the working implementation for an unquantized checkpoint, so nothing
    here may assume a quantizer exists.
    """

    #: Name reported in logs and the decision table.
    format = "dense"
    #: ``quant_method`` values from a checkpoint's quantization_config that select this backend.
    quant_methods = ()

    def __init__(self, config=None, caps=None, model=None, hf_quantizer=None, compute_dtype=None,
                 device=None, shape_adopter=None):
        self.config = config
        self.model = model
        self.hf_quantizer = hf_quantizer
        self.device = device
        self._caps = caps
        self._compute_dtype = compute_dtype
        # The engine reconciles meta placeholders whose shape disagrees with the checkpoint. That
        # is the model class's business rather than the format's, so it is injected rather than
        # reimplemented here.
        self._shape_adopter = shape_adopter
        self._config_dict = _as_dict(config)
        self._declared_bits = _find_int(self._config_dict, BIT_WIDTH_KEYS)

    # -- detection -------------------------------------------------------------------------------

    @classmethod
    def matches(cls, quant_method, config):
        return quant_method in cls.quant_methods

    @classmethod
    def example_config(cls):
        """The smallest quantization_config that would select this backend.

        Only the decision table uses it: it reports what each format *would* do here, which means
        building one backend per format without a checkpoint to hand.
        """
        return {"quant_method": cls.quant_methods[0]} if cls.quant_methods else None

    # -- context ---------------------------------------------------------------------------------

    @property
    def caps(self):
        """The running device's capabilities. Resolved late so tests can substitute them."""
        if self._caps is None:
            self._caps = get_caps(self.device, announce=False)
        return self._caps

    @property
    def compute_dtype(self):
        return self._compute_dtype if self._compute_dtype is not None else self.caps.compute_dtype

    # -- layout ----------------------------------------------------------------------------------

    #: Suffixes that mark the payload itself rather than something describing it. A format whose
    #: weight is never stored under its own name declares them, so the payload is still found.
    PAYLOAD_SUFFIXES = ("_packed",)

    def logical_name(self, tensor_name, known=()):
        """The weight a checkpoint tensor belongs to.

        Companions are recognised by their suffix rather than by a per-architecture table, so a
        checkpoint format nobody has shipped yet still groups correctly as long as it follows the
        convention every current one follows. `known` is the rest of the shard, for formats whose
        grouping is structural rather than by name.
        """
        for marker in BNB_MARKERS:
            if marker in tensor_name:
                return tensor_name.split(marker)[0]
        module, _, leaf = tensor_name.rpartition(".")
        for suffix in COMPANION_SUFFIXES:
            if leaf.endswith(suffix) and leaf != suffix:
                stem = leaf[: -len(suffix)]
                return f"{module}.{stem}" if module else stem
        return tensor_name

    def is_payload(self, tensor_name, logical_name):
        """Whether this tensor is the weight itself rather than something describing it."""
        if tensor_name == logical_name:
            return True
        return any(tensor_name == logical_name + suffix for suffix in self.PAYLOAD_SUFFIXES)

    def is_consumed(self, tensor_name, logical_name):
        """Whether this tensor is folded into its weight rather than placed as a parameter.

        bitsandbytes stores a quant state under the weight it belongs to and reads it back out of
        the shard when it reconstructs the parameter. The module never grows a parameter for it, so
        placing it would fail -- it is spanned and counted, and that is all.
        """
        return any(marker in tensor_name for marker in BNB_MARKERS)

    def plan(self, state_dict):
        """Group a shard's tensors into the logical weights they make up.

        Order is first-appearance, and within a weight the payload comes first, so a caller walking
        the plan sees the shard in the order it was written.
        """
        known = set(state_dict)
        groups = OrderedDict()
        for tensor_name in state_dict:
            groups.setdefault(self.logical_name(tensor_name, known), []).append(tensor_name)

        weights = []
        for logical_name, names in groups.items():
            payload = next((n for n in names if self.is_payload(n, logical_name)), names[0])
            ordered = [payload] + [n for n in names if n != payload]
            specs = [TensorSpec.of(n, state_dict[n]) for n in ordered]
            values = {n: state_dict[n] for n in ordered}
            weights.append(PackedWeight(logical_name, specs, self, values=values, shard=state_dict))
        return weights

    def plan_from_specs(self, specs):
        """The same grouping over checkpoint metadata, for sizing a layer without reading it."""
        known = {spec.name for spec in specs}
        groups = OrderedDict()
        for spec in specs:
            groups.setdefault(self.logical_name(spec.name, known), []).append(spec)

        weights = []
        for logical_name, group in groups.items():
            payload = next((s for s in group if self.is_payload(s.name, logical_name)), group[0])
            ordered = [payload] + [s for s in group if s is not payload]
            weights.append(PackedWeight(logical_name, ordered, self))
        return weights

    # -- sizing ----------------------------------------------------------------------------------

    def declared_bits(self):
        """The width the checkpoint declares for its quantized weights, or ``None``."""
        return self._declared_bits

    def bits(self, weight):
        """How many bits one logical value of this weight occupies as stored.

        A checkpoint declares one width, but it does not apply to everything in the file: a bias
        and a layer norm sit next to a 4-bit weight and are still 16 bits wide. So a payload that
        is already a plain high-precision float answers for itself, and the declaration is only
        consulted for the packed ones. Without any declaration the payload's own dtype is the
        answer, which is right for an unquantized weight and for fp8.
        """
        payload = weight.payload
        stored = payload.itemsize * 8
        if payload.is_floating_point and stored >= 16:
            return stored
        return self.declared_bits() or stored

    def values_per_item(self, weight):
        """How many logical values one payload element holds.

        Packing is always "fit as many n-bit values as the storage element has room for": four-bit
        GPTQ packs eight into an int32, MXFP4 two into a uint8, an fp8 weight one into a byte. So
        the factor falls out of the two widths and no format needs a constant of its own.
        """
        stored = weight.payload.itemsize * 8
        bits = self.bits(weight)
        return max(1, stored // bits) if bits else 1

    def logical_shape(self, weight):
        """The shape the module ends up with, where the checkpoint says so outright.

        compressed-tensors ships a ``weight_shape`` tensor precisely because a packed payload's
        shape does not reveal it. When it is there it is authoritative; otherwise the count is
        recovered from the packing factor and the shape stays unknown.
        """
        for spec in weight.companions:
            if spec.name.endswith("_shape") and weight.has_values:
                try:
                    return tuple(int(v) for v in weight.value_of(spec.name).flatten().tolist())
                except Exception:  # noqa: BLE001 - a malformed hint must not sink the load
                    return None
        return None

    def logical_numel(self, weight):
        shape = weight.logical_shape
        if shape is not None:
            count = 1
            for dim in shape:
                count *= dim
            return count
        return weight.payload.numel * self.values_per_item(weight)

    def expanded_bytes(self, weight):
        return self.logical_numel(weight) * _itemsize(self.compute_dtype)

    # -- the per-device decision -----------------------------------------------------------------

    def needs_scratch(self, weight):
        """An unquantized weight is already in a computable form; only sub-16-bit storage is not.

        fp8 is the one case a plain checkpoint can still hit: the payload is a byte wide, and a
        device without fp8 arithmetic has to widen it before it can multiply anything by it.
        """
        bits = self.bits(weight)
        if bits >= 16:
            return False
        if bits == 8 and weight.payload.is_floating_point:
            return not self.caps.supports_fp8
        return True

    # -- placement -------------------------------------------------------------------------------

    def needs_quantizer(self, tensor_name):
        """Whether the quantizer reconstructs this parameter instead of the loader placing it."""
        quantizer = self.hf_quantizer
        if quantizer is None:
            return False
        # transformers renamed check_quantized_param -> param_needs_quantization.
        if hasattr(quantizer, "param_needs_quantization"):
            return bool(quantizer.param_needs_quantization(self.model, tensor_name))
        return bool(quantizer.check_quantized_param(self.model, param_value=None,
                                                    param_name=tensor_name, state_dict={}))

    def load_verbatim(self, tensor_name, value):
        """Whether a checkpoint tensor must be placed without a dtype cast.

        Casting a pre-quantized payload to the runtime dtype destroys it: an fp8 weight loses its
        quantization, and a packed 4-bit tensor (stored as packed integers) becomes meaningless
        floats. Equally important for very large models, widening on load multiplies a layer's
        footprint by ~4x, which is exactly what keeps the largest checkpoints from fitting. So
        anything that is not a plain high-precision float is kept as it is.
        """
        if not value.is_floating_point():
            # Packed 4-bit payloads, zero points, g_idx, shape metadata.
            return True
        if value.element_size() == 1:
            # Any 8-bit float: fp8 e4m3/e5m2 weights, and the e8m0 scales MXFP4 uses.
            return True
        return tensor_name.endswith(COMPANION_SUFFIXES)

    def target_dtype(self, tensor_name, value):
        """The dtype this tensor lands in on the device."""
        return value.dtype if self.load_verbatim(tensor_name, value) else self.compute_dtype

    def adopt_shape(self, tensor_name, value):
        if self._shape_adopter is not None:
            self._shape_adopter(tensor_name, value)

    def place(self, tensor_name, value, device, shard=None):
        """Put one checkpoint tensor where the module expects it."""
        if self.needs_quantizer(tensor_name):
            # Schemes that reconstruct a parameter (bitsandbytes, fp8, run-compressed CT) build it
            # from the payload plus companion quant-state tensors carried in the shard, and do
            # their own placement.
            self.hf_quantizer.create_quantized_param(self.model, value, tensor_name, device,
                                                     shard if shard is not None else {})
            return tensor_name

        self.adopt_shape(tensor_name, value)
        if self.load_verbatim(tensor_name, value):
            set_module_tensor_to_device(self.model, tensor_name, device, value=value)
        else:
            set_module_tensor_to_device(self.model, tensor_name, device, value=value,
                                        dtype=self.compute_dtype)
        return tensor_name

    def prepare_layer(self, state_dict):
        """Whatever has to happen to a shard before its tensors can be placed.

        Nothing, for a format whose modules read what the checkpoint stores. Formats that ship a
        payload no module can consume override this and expand it here.
        """
        return state_dict

    # -- reporting -------------------------------------------------------------------------------

    def decision(self):
        """What this backend does on the running device, and which query decided it."""
        return {
            "format": self.format,
            "capability": "none: stored weights are already in a computable dtype",
            "available": True,
            "path": "as_stored",
            "reason": "an unquantized checkpoint is placed as it is, cast to the compute dtype",
        }

    def summary(self):
        info = dict(self.decision())
        info.update({
            "quant_method": quant_method_of(self.config),
            "declared_bits": self._declared_bits,
            "compute_dtype": str(self.compute_dtype).replace("torch.", ""),
            "hf_quantizer": type(self.hf_quantizer).__name__ if self.hf_quantizer else None,
        })
        return info

    def __repr__(self):
        return f"<{type(self).__name__} {self.format}>"


# -------------------------------------------------------------------------------------------------
# config inspection
# -------------------------------------------------------------------------------------------------

def _as_dict(config):
    """A checkpoint's quantization_config as a plain dict, whatever shape it arrived in."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    for attr in ("to_dict", "to_diff_dict", "dict"):
        method = getattr(config, attr, None)
        if callable(method):
            try:
                result = method()
                if isinstance(result, dict):
                    return result
            except Exception:  # noqa: BLE001 - a config we cannot serialise is still usable
                pass
    return {k: v for k, v in vars(config).items() if not k.startswith("_")} if hasattr(
        config, "__dict__") else {}


def _find_int(node, keys, depth=0):
    """First integer stored under any of `keys`, at any depth.

    compressed-tensors records the width per config group rather than at the top level, and the
    group names are the checkpoint author's choice, so the search cannot be a fixed path.
    """
    if depth > 6 or not isinstance(node, (dict, list, tuple)):
        return None
    if isinstance(node, dict):
        for key in keys:
            value = node.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return int(value)
        children = node.values()
    else:
        children = node
    for child in children:
        found = _find_int(child, keys, depth + 1)
        if found is not None:
            return found
    return None


def quant_method_of(quantization_config):
    """The ``quant_method`` a checkpoint declares, lowercased, or ``None`` if it declares none."""
    if quantization_config is None:
        return None
    method = None
    if isinstance(quantization_config, dict):
        method = quantization_config.get("quant_method")
    else:
        method = getattr(quantization_config, "quant_method", None)
        if method is None:
            method = _as_dict(quantization_config).get("quant_method")
    # transformers models it as an enum on some versions and a plain string on others.
    method = getattr(method, "value", method)
    return str(method).lower() if method is not None else None


# -------------------------------------------------------------------------------------------------
# the registry
# -------------------------------------------------------------------------------------------------

_BACKENDS = []


def register_backend(cls):
    """Register a backend for detection. First match wins, so register specific ones first."""
    if cls not in _BACKENDS:
        _BACKENDS.append(cls)
    return cls


def registered_backends():
    _load_backends()
    return tuple(_BACKENDS)


def _load_backends():
    """Import the concrete backends so they are registered.

    Done lazily rather than at module import because the backends import this module for the base
    class, and a top-level import here would close the circle.
    """
    if not _BACKENDS:
        from . import safetensors_quant  # noqa: F401 - imported for its registrations


def detect_backend(quantization_config=None, **context):
    """Pick the backend for a checkpoint from what the checkpoint says about itself.

    An unrecognised ``quant_method`` is not a failure. transformers may still have wired up a
    quantizer for it, and delegating to that quantizer is what every format-specific backend does
    anyway, so an unknown format degrades to the generic delegating path rather than refusing to
    load. Only a checkpoint that declares nothing at all is treated as unquantized.
    """
    _load_backends()
    method = quant_method_of(quantization_config)
    for cls in _BACKENDS:
        if cls.matches(method, quantization_config):
            return cls(config=quantization_config, **context)

    from .safetensors_quant import HfQuantizerBackend
    if method is not None:
        log.info("quantization method %r is not one this build knows by name; loading it through "
                 "the quantizer transformers wired up for it", method)
        return HfQuantizerBackend(config=quantization_config, **context)
    return QuantBackend(config=quantization_config, **context)


def decision_table(caps=None, compute_dtype=None):
    """What every known format would do on this machine, and which query decided it.

    Reported at load and in bug reports, because "it was slow" and "it dequantized every layer
    because your build has no fused kernel" are the same observation and only one of them is
    actionable.
    """
    _load_backends()
    rows = []
    for cls in list(_BACKENDS) + [QuantBackend]:
        backend = cls(config=cls.example_config(), caps=caps, compute_dtype=compute_dtype)
        rows.append(backend.decision())
    return rows
