
import logging
import os
from collections import OrderedDict
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation import GenerationMixin
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer

from .hw import caps
from .hw.caps import get_caps
from .hw.profile import HardwareProfile
from .quant import detect_backend
from .quant.safetensors_quant import announce_backend
from .memory import (CLASS_ALWAYS, KIND_DENSE, PinCandidate, TieredWeightCache, expert_kind,
                     is_expert, pin_budget_from, plan_pins)
from .moe import (LAYOUT_MODULE_LIST, ExpertContainer, ExpertLayout, RouterSelection,
                  detect_expert_layout, resolve_top_k, summarize as summarize_experts)
from .streaming import HostStagingPool, LayerLoader, WeightTransfer
from .profiler import LayeredProfiler

from .utils import clean_memory, load_layer, load_layer_rows, load_layer_subset, \
    find_or_create_local_splitted_path, reject_compression_argument
from .persist import ModelPersister


# Helpers that transformers 5.0 moved out of transformers.utils.generic. Remote model code is
# routinely written against an older transformers and still imports them from the old location,
# which makes such models fail to import at all. Re-exporting is enough to load them.
_RELOCATED_TRANSFORMERS_SYMBOLS = {
    'OutputRecorder': 'transformers.utils.output_capturing',
    'check_model_inputs': 'transformers.utils.output_capturing',
}


def restore_relocated_transformers_symbols():
    """Re-export moved transformers helpers under their old names, where they are missing."""
    import importlib
    import transformers.utils.generic as generic

    for name, new_home in _RELOCATED_TRANSFORMERS_SYMBOLS.items():
        if hasattr(generic, name):
            continue
        try:
            module = importlib.import_module(new_home)
        except ImportError:
            continue
        symbol = getattr(module, name, None)
        if symbol is not None:
            setattr(generic, name, symbol)


class _ResidentModule:
    """One streamed unit's weights, wherever they currently are.

    A unit is a dense module or a single MoE expert -- the cache keys both the same way, so this
    carries the cache key rather than a layer index.

    `host` is the CPU-side state dict and `moved` is the list of parameter names currently bound on
    the device. Both, one, or neither may be set: that is what the cache's tiers mean here.
    """

    __slots__ = ("key", "host", "moved")

    def __init__(self, key, host=None, moved=None):
        self.key = key
        self.host = host
        self.moved = moved


class RocketModel:
    """
    Memory-frugal wrapper around a Hugging Face ``*ForCausalLM`` model.

    The checkpoint is split into per-layer shards on disk. The real transformers model is
    instantiated on the ``meta`` device (no memory used) and owns the full forward / generation
    logic. RocketLLM only attaches forward hooks to each big module (embeddings, every decoder
    layer, the final norm and the lm_head) to stream that module's weights disk -> GPU right
    before it runs and free them right after, prefetching the next module on a worker thread.

    Because transformers drives the forward pass, RocketLLM no longer needs to track per-architecture
    attention/rotary/cache details: new model architectures work as soon as transformers supports
    them.
    """

    # Subclasses override this to point at non-standard module names.
    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'model.embed_tokens',
                                 'layer_prefix': 'model.layers',
                                 'norm': 'model.norm',
                                 'lm_head': 'lm_head'}

    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=None, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 vram_reserve=None, host_cache_gb=None, io_workers=None, window_max=None,
                 pin_policy="auto"):
        """
        Parameters
        ----------
        model_local_path_or_repo_id : str or Path
            path to the local model checkpoint or huggingface repo id
        device : str, optional
            device, by default "cuda:0"
        dtype : torch.dtype, optional
            runtime dtype; defaults to the model's own config.torch_dtype (usually bfloat16 for
            modern models). float16 has too narrow a range for very deep models and overflows to
            inf/NaN, which silently corrupts the output, so we don't force it.
        max_seq_len : int, optional
            max seq length, by default 512
        layer_shards_saving_path : str, optional
            optional path to save the splitted shards, by default next to the model cache
        profiling_mode : bool, optional
            whether to profile the model loading time, default False
        compression: str, optional
            removed. RocketLLM imports pre-quantized checkpoints and does not quantize models
            itself; passing anything here raises with the list of formats it does read.
        hf_token: str, optional
            huggingface api token
        prefetching: bool, optional
            overlap the next layers' disk reads with the current layer's compute
        delete_original: bool, optional
            delete the original downloaded checkpoint after splitting to save disk space

        The five options below are all DEBUGGING OVERRIDES. Every one of them defaults to None,
        meaning "take the value the HardwareProfile measured for this machine", and that is the
        setting you want. RocketLLM has no reference machine: a number that is right on the box it
        was chosen on is wrong on the next one, so these exist to reproduce a problem or to bisect
        a suspected bad measurement, not to tune a healthy run.

        vram_reserve: int, optional
            bytes of device memory held back for activations, workspace and fragmentation.
            Default: profile `reserve_bytes`, built from the allocator's measured fragmentation
            ratio and workspace high-water mark.
        host_cache_gb: float, optional
            gigabytes of host RAM the cache may hold as its middle tier. Default: profile
            `host_cache_bytes`, a share of *available* RAM after OS headroom. Zero is valid and
            means evictions drop straight to storage.
        io_workers: int, optional
            concurrent storage readers. Default: profile `io_workers`, the concurrency that was
            measured to saturate this machine's storage -- one reader is latency-bound below a fast
            drive's rated bandwidth, too many thrash a slow one.
        window_max: int, optional
            hard cap on how many decoder layers the prefetch window may hold. Default: profile
            `window_budget_bytes` divided by the largest layer, which is the memory-derived answer.
        pin_policy: str, optional
            "auto" (default) ranks candidates by access-frequency-per-packed-byte and fills the pin
            budget; "off" pins nothing and streams everything, which is the pure-streaming
            configuration and is what a device with no spare memory gets anyway.
        """

        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()

        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.hf_quantizer = None

        reject_compression_argument(compression)

        self.hf_token = hf_token

        restore_relocated_transformers_symbols()

        self.set_layer_names_dict()

        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(
            model_local_path_or_repo_id,
            layer_shards_saving_path,
            layer_names=self.layer_names_dict,
            hf_token=hf_token,
            delete_original=delete_original)

        self.running_device = device
        self.device = torch.device(self.running_device)

        # Prefer transformers' native implementation; only trust the model's bundled remote code when
        # transformers doesn't recognize the architecture. Vendored remote code is frequently pinned
        # to an old transformers and breaks against the current cache/generation APIs (e.g.
        # DeepSeek-V2's modeling_deepseek.py calls the long-removed DynamicCache.seen_tokens).
        token_kwargs = {'token': hf_token} if hf_token is not None else {}
        try:
            self.config = AutoConfig.from_pretrained(
                self.model_local_path, trust_remote_code=False, **token_kwargs)
            self.trust_remote_code = False
        except Exception:
            self.config = AutoConfig.from_pretrained(
                self.model_local_path, trust_remote_code=True, **token_kwargs)
            self.trust_remote_code = True

        # Default to the model's native dtype (bf16 for most modern models). Forcing fp16 overflows
        # on deep models (e.g. Qwen3-235B's 94 layers) and produces garbage; bf16's wider range
        # avoids it. Users can still override via dtype=.
        #
        # The choice then goes through the device abstraction, which is what knows whether this
        # hardware can actually run it. A checkpoint asking for bf16 on a card without bf16 used
        # to be honoured as written; now it degrades to fp16 and says so once, because the failure
        # mode otherwise is silently wrong tokens rather than an error.
        self.caps = get_caps(self.running_device)
        if dtype is None:
            cfg_dtype = getattr(self.config, "torch_dtype", None)
            if isinstance(cfg_dtype, str):
                cfg_dtype = getattr(torch, cfg_dtype, None)
            requested = cfg_dtype if isinstance(cfg_dtype, torch.dtype) else torch.float16
            dtype = self.caps.select_compute_dtype(requested)
        self.running_dtype = dtype
        self.dtype = self.running_dtype

        self.generation_config = self.get_generation_config()
        self.tokenizer = self.get_tokenizer(hf_token=hf_token)

        self.prefetching = prefetching
        # Debugging overrides; None everywhere means "use what the machine measured".
        self._overrides = {
            "reserve_bytes": vram_reserve,
            "host_cache_bytes": None if host_cache_gb is None else int(host_cache_gb * 1024 ** 3),
            "io_workers": io_workers,
            "window_max": window_max,
        }
        if pin_policy not in ("auto", "off"):
            raise ValueError(f"pin_policy must be 'auto' or 'off', not {pin_policy!r}")
        self.pin_policy = pin_policy
        self._layer_bytes = {}

        # Staging buffers are sized from the machine, not from a constant. Probing is cached by
        # hardware fingerprint, so this is a one-off cost; if it fails for any reason the pool
        # runs with a zero budget, which means no pooling and no pinning -- slower, still correct.
        self.staging_pool = HostStagingPool(self.caps, self._staging_budget_bytes())
        # The copy stream, and the thing that owns a staged buffer's lifetime until its event
        # fires. Built here because the pool has to know how to ask it for finished transfers
        # before it decides it has no free buffer.
        self.transfer = WeightTransfer(self.caps, pool=self.staging_pool)
        # Header-only reads for sizing, and the byte-range reads the cache's storage tier uses.
        self.loader = LayerLoader(self.checkpoint_path, self.staging_pool, profile=self.profile,
                                  io_workers=self._overrides.get("io_workers"))

        self.init_model()

        # compute layer count from the instantiated model
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        layers_count = len(model_attr)

        self.layer_names = [self.layer_names_dict['embed']] + \
                           [f'{self.layer_names_dict["layer_prefix"]}.{i}' for i in range(layers_count)] + \
                           [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']]

        self.max_seq_len = max_seq_len

        self.set_layers_from_layer_names()
        self._load_resident_modules()
        self._install_streaming_hooks()
        self._build_cache()

    # ---- customization hooks for subclasses -------------------------------------------------

    def get_generation_config(self):
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig()

    def get_tokenizer(self, hf_token=None):
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

    # ---- model construction -----------------------------------------------------------------

    def _propagate_attn_implementation(self, impl):
        """Push the attention choice down into nested sub-configs.

        Multimodal wrappers keep the real decoder under a sub-config -- Kimi K3 uses ``text_config``
        -- and transformers records the request only on the config it was handed. The sub-model then
        reads an unset value and falls through to a flash-attention path, which fails outright on a
        machine without flash-attn installed.
        """
        from transformers import PretrainedConfig

        def walk(cfg, depth=0):
            if depth > 2:
                return
            for sub in vars(cfg).values():
                if isinstance(sub, PretrainedConfig):
                    sub._attn_implementation = impl
                    walk(sub, depth + 1)

        walk(self.config)

    def init_model(self):
        # Build the real model on meta (no memory). include_buffers=False so non-persistent
        # buffers such as rotary inv_freq are actually computed (they aren't in the checkpoint).
        self.model = None
        try:
            self._propagate_attn_implementation("sdpa")
            with init_empty_weights(include_buffers=False):
                self.model = AutoModelForCausalLM.from_config(
                    self.config, attn_implementation="sdpa", trust_remote_code=self.trust_remote_code)
        except (ValueError, TypeError) as e:
            print(f"attn_implementation='sdpa' not available ({e}), falling back to eager attention")
            self.model = None
        if self.model is None:
            # Some (often remote-code) architectures don't support sdpa and also default to it, so we
            # must request eager explicitly; otherwise transformers re-selects sdpa and errors again.
            self._propagate_attn_implementation("eager")
            with init_empty_weights(include_buffers=False):
                self.model = AutoModelForCausalLM.from_config(
                    self.config, attn_implementation="eager", trust_remote_code=self.trust_remote_code)

        quantization_config = getattr(self.config, "quantization_config", None)
        if quantization_config is None:
            # Nested multimodal configs (Kimi K3) keep it under text_config.
            quantization_config = getattr(getattr(self.config, "text_config", None),
                                          "quantization_config", None)
        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)
            # compressed-tensors registers a hook that expands every packed module on the first
            # forward. That undoes per-expert streaming (a K3 layer becomes ~56GB) and is also
            # unnecessary: we decompress each expert ourselves as it loads. Remove the hook.
            hook = getattr(self.model, "ct_decompress_hook", None)
            if hook is not None:
                hook.remove()
                delattr(self.model, "ct_decompress_hook")

        # Which checkpoint format this is, and what it costs to put on *this* device, is the quant
        # package's business from here on. The model no longer inspects tensor names to work out
        # what it is holding: it asks for the layer's logical weights and places what it is given.
        self.quant = detect_backend(quantization_config,
                                    caps=self.caps,
                                    model=self.model,
                                    hf_quantizer=self.hf_quantizer,
                                    compute_dtype=self.running_dtype,
                                    device=self.running_device,
                                    shape_adopter=self._adopt_checkpoint_shape)
        if quantization_config is not None:
            announce_backend(self.quant)

        self.model.eval()
        self.model.tie_weights()
        self.model.generation_config = self.generation_config

        # Move all (already-materialized) buffers to the running device, preserving their dtype.
        # This includes rotary inv_freq, which transformers computes once at the model level and
        # passes down to every decoder layer.
        for buffer_name, buffer in self.model.named_buffers():
            if buffer is not None and buffer.device.type != 'meta':
                set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer)

        # Force the model to report the running (cuda) device even though its parameters live on
        # meta between layer executions, so transformers' generation utilities place inputs/cache
        # tensors on the right device.
        self._patch_device_property()

    def _patch_device_property(self):
        running_device = torch.device(self.running_device)
        running_dtype = self.running_dtype
        base_cls = type(self.model)

        # transformers >= 4.50 removed GenerationMixin from PreTrainedModel, so model classes that
        # predate that change (or ship as remote code, like Kimi K3's multimodal wrapper) no longer
        # have .generate(). They still define prepare_inputs_for_generation, so mixing the class
        # back in restores generation. It must come after the model class, per transformers.
        extra_bases = () if isinstance(self.model, GenerationMixin) else (GenerationMixin,)
        if extra_bases:
            print(f"{base_cls.__name__} does not inherit GenerationMixin; mixing it in so "
                  f"generate() works.")

        class _RocketRuntimeModel(base_cls, *extra_bases):
            @property
            def device(self):
                return running_device

            @property
            def dtype(self):
                return running_dtype

        self.model.__class__ = _RocketRuntimeModel

    def set_layers_from_layer_names(self):
        self.layers = []

        model_attr = self.model
        for attr_name in self.layer_names_dict["embed"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.extend(list(model_attr))

        model_attr = self.model
        for attr_name in self.layer_names_dict["norm"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["lm_head"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

    # ---- weight streaming -------------------------------------------------------------------

    def load_layer_to_cpu(self, layer_name):
        t = time.time()
        state_dict = load_layer(self.checkpoint_path, layer_name)
        if self.profiling_mode:
            self.profiler.add_profiling_time('load_safe_tensor', time.time() - t)

        # These tensors are no longer what crosses the link: move_layer_to_device packs them into
        # one staging buffer and transfers that. Pinning each of them here would page-lock the
        # whole layer twice over -- once as sixty separate driver calls, once as the buffer -- to
        # speed up a copy that now never happens. Only the staging buffer is worth pinning.
        return state_dict

    def _prepare_layer(self, state_dict):
        """Whatever the checkpoint's format has to do to a shard before its tensors can be placed.

        Nothing at all for most formats. For the packed ones it is the dequantization, which runs
        on the device after the *packed* bytes have crossed the link -- so this is the seam the
        benchmark charges its dequant phase to.
        """
        return self.quant.prepare_layer(state_dict)

    def _adopt_checkpoint_shape(self, param_name, value):
        """Resize a meta placeholder whose shape disagrees with the checkpoint.

        A model class builds its parameters from the config, and that construction can disagree
        with the weights actually shipped. Kimi K3 sizes ``A_log`` from ``num_heads`` while its
        checkpoint stores one entry per head channel, which would abort the load. For a parameter
        still on meta -- i.e. one we have never materialised -- the checkpoint is the source of
        truth, so adopt its shape.
        """
        module_path, _, attr = param_name.rpartition('.')
        try:
            module = self.model.get_submodule(module_path) if module_path else self.model
        except AttributeError:
            return
        current = module._parameters.get(attr)
        if current is None or current.device.type != 'meta' or current.shape == value.shape:
            return

        if not hasattr(self, '_shape_adoption_warned'):
            self._shape_adoption_warned = set()
        if attr not in self._shape_adoption_warned:
            self._shape_adoption_warned.add(attr)
            print(f"{attr}: checkpoint ships {tuple(value.shape)} but the model class builds "
                  f"{tuple(current.shape)}; using the checkpoint shape.")

        module.register_parameter(
            attr,
            torch.nn.Parameter(torch.empty(value.shape, device='meta', dtype=current.dtype),
                               requires_grad=False),
        )

    def move_layer_to_device(self, state_dict):
        """Place a layer's weights on the device, in as few transfers as possible.

        A decoder layer is on the order of sixty separate tensors. Sent one at a time, none of them
        is large enough to reach link bandwidth, and the per-transfer overhead is paid sixty times.
        So tensors that can share a buffer are packed into one contiguous staging buffer, sent in a
        single transfer, and then bound as views into the device buffer -- no second copy, and one
        allocation for the whole layer.

        Grouping is by target dtype, which keeps the arithmetic obvious and still collapses a
        typical all-bf16 layer to a single transfer. Anything that cannot be packed -- a param the
        quantizer reconstructs itself, or one that errors on the way in -- falls back to being
        placed on its own, because a slower correct path beats a faster broken one.
        """
        state_dict = self._prepare_layer(state_dict)

        groups, individual = self._plan_transfer(state_dict)

        moved = []
        for target_dtype, entries in groups.items():
            if len(entries) < 2:
                # One tensor is not a coalesce; packing it would just add a host-side copy.
                individual.extend((weight, name) for weight, name, _ in entries)
                continue
            try:
                moved.extend(self._move_coalesced([(name, value) for _, name, value in entries],
                                                  target_dtype))
            except Exception as exc:  # noqa: BLE001 - correctness over throughput
                caps.announce_once(
                    "coalesce-fallback",
                    f"could not coalesce a layer's transfers ({exc}); falling back to placing "
                    f"tensors individually, which is slower but produces the same result.",
                    logging.INFO)
                individual.extend((weight, name) for weight, name, _ in entries)

        for weight, param_name in individual:
            weight.place(param_name, self.running_device, shard=state_dict)
            moved.append(param_name)
        return moved

    def _plan_transfer(self, state_dict):
        """Split a layer's weights into coalescable groups and ones that must go on their own.

        The split is the registry's to make, not this class's: which tensors are one logical weight,
        and which of them the quantizer reconstructs rather than the loader placing, is a property
        of the checkpoint's format. What is decided here is only how to move what comes back.
        """
        groups = OrderedDict()
        individual = []
        for weight in self.quant.plan(state_dict):
            for param_name, value, target in weight.placements():
                self._adopt_checkpoint_shape(param_name, value)
                groups.setdefault(target, []).append((weight, param_name, value))
            # The quantizer reconstructs these from the payload plus companion quant-state tensors
            # and does its own placement; there is no plain buffer to pack.
            individual.extend((weight, name) for name in weight.quantizer_names())
        return groups, individual

    def _move_coalesced(self, entries, target_dtype):
        """Pack, send once, then bind each parameter as a view into the device buffer.

        The transfer is issued on a dedicated copy stream and the compute stream is ordered behind
        it by an event, rather than the CPU waiting for the copy to land. Nothing here blocks: the
        binds below and the forward that follows are queued behind the event, so the copy overlaps
        whatever the device is still working on. The host buffer is leased, not borrowed, and the
        transfer layer holds that lease until its event has actually fired -- releasing it here
        would let the next layer overwrite bytes still in flight.
        """
        total = sum(value.numel() for _, value in entries)
        lease = self.staging_pool.lease(total, target_dtype)
        host = lease.view

        offset = 0
        for _, value in entries:
            count = value.numel()
            # copy_ casts on the way in, so the transfer carries the runtime dtype, not the
            # checkpoint's -- fewer bytes over the link when the checkpoint is wider.
            host.narrow(0, offset, count).copy_(value.reshape(-1))
            offset += count

        device_buffer = self.transfer.send_buffer(host, lease).resolve()

        moved = []
        offset = 0
        for param_name, value in entries:
            count = value.numel()
            view = device_buffer.narrow(0, offset, count).view(value.shape)
            set_module_tensor_to_device(self.model, param_name, self.running_device, value=view)
            offset += count
            moved.append(param_name)
        return moved

    def _staging_buffer(self, count, dtype):
        """A flat host buffer to pack a layer into, reused across layers.

        The pool hands the same page-locked buffer back for layers of the same size class, which
        is what makes pinning affordable: allocated per layer it costs more than the transfer it
        accelerates, allocated once it costs nothing per token.
        """
        return self.staging_pool.buffer(count, dtype)

    def _load_resident_modules(self):
        """Load modules that sit outside the streamed embed -> layers -> norm -> lm_head sequence.

        Multimodal checkpoints carry a vision tower and projector, and some architectures add
        extra top-level norms. They never get a streaming hook, so without this they would stay on
        the meta device and fail the moment they run. They are small (well under a GB), so we load
        them once and leave them resident.
        """
        for name in self.layer_names_dict.get('resident', []):
            try:
                state_dict = self.load_layer_to_cpu(name)
            except FileNotFoundError:
                # Not every checkpoint of a given architecture ships every optional module.
                continue
            self.move_layer_to_device(state_dict)

    def _install_streaming_hooks(self):
        # Modules execute in this order during a forward: embed -> layers -> norm -> lm_head.
        n = len(self.layer_names)

        # Detect tied input/output embeddings. When tied, lm_head shares the embedding weight, so
        # there is no separate lm_head shard. We keep the embedding resident on the GPU (it is the
        # only copy and such models are small) and re-tie lm_head to it, then stream only the
        # decoder layers and the final norm.
        self.tie_word_embeddings = bool(getattr(self.config, "tie_word_embeddings", False))

        if self.tie_word_embeddings:
            embed_state = self.load_layer_to_cpu(self.layer_names[0])
            self.move_layer_to_device(embed_state)
            self.model.tie_weights()
            self._streamed_indices = list(range(1, n - 1))  # decoder layers + final norm
        else:
            self._streamed_indices = list(range(n))

        self._streamed_set = set(self._streamed_indices)

        self._setup_expert_streaming()

        for idx in self._streamed_indices:
            module = self.layers[idx]
            module._rocketllm_idx = idx
            module.register_forward_pre_hook(self._pre_hook)
            module.register_forward_hook(self._post_hook)

    # ---- MoE expert streaming ---------------------------------------------------------------

    def _setup_expert_streaming(self):
        """Stream individual MoE experts instead of whole decoder layers, wherever a layer has any.

        A sparse MoE layer holds tens to hundreds of experts and routes each token to a handful, so
        materialising the layer moves one or two orders of magnitude more bytes than the token
        reads: a Kimi K3 layer's experts are ~55GB expanded, of which a token touches ~1GB.

        Which layers those are is worked out from structure -- see rocketllm.moe.detect -- rather
        than from a subclass declaring `expert_prefix`. That declaration was the old gate, and it
        meant exactly one architecture got this and every other mixture streamed whole layers. It
        survives as a manual override for a checkpoint whose experts detection cannot see.

        Both layouts land here, and what each one saves differs.

        A list of expert modules gets a forward hook per expert, and each expert becomes a cache
        entry in its own right. That is the saving which holds unconditionally: residency is decided
        per expert against the device budget instead of a layer being kept or dropped whole, so a
        budget too small to hold one layer still holds a useful number of that layer's experts,
        where whole-layer caching would keep nothing at all. Whether the *read* also drops is the
        model's decision rather than ours -- an implementation that walks every expert and masks the
        unrouted ones (transformers 4.51's Mixtral and Qwen2-MoE do exactly this) still touches them
        all, while one that skips them reads only the routed few. The engine loads whatever runs.

        A fused ``[num_experts, ...]`` tensor has no per-expert module to hook, so its container is
        hooked instead and reads only the rows the router just chose. There the read always drops,
        because the selection is intercepted rather than inferred from which modules get called.
        """
        self._expert_streaming = False
        self._non_expert_keys = {}
        self._expert_layouts = {}
        #: Cache key -> the checkpoint tensors that expert is made of, and what they cost packed.
        self._expert_tensor_keys = {}
        self._expert_byte_counts = {}

        # Both layouts read individual tensors -- or individual rows of one -- out of a shard, which
        # is a safetensors capability. Other persisters store something this cannot seek into.
        if type(ModelPersister.get_model_persister()).__name__ != 'SafetensorModelPersister':
            return

        layer_prefix = self.layer_names_dict['layer_prefix']
        hooked_experts = 0
        fused_containers = 0
        skipped = []

        for idx in self._streamed_indices:
            layer_name = self.layer_names[idx]
            if not layer_name.startswith(layer_prefix + '.'):
                continue
            try:
                shapes, packed_bytes = self._layer_tensor_specs(layer_name)
            except Exception:  # noqa: BLE001 - a layer we cannot inspect simply streams whole
                continue

            layout = detect_expert_layout(self.layers[idx], shapes, layer_name, self.config)
            skipped.extend(layout.skipped)
            containers = list(layout.containers) or self._configured_containers(shapes)

            accepted = []
            for container in containers:
                reason = self._cannot_stream_alone(container)
                if reason is not None:
                    skipped.append((container.path, reason))
                    continue
                if container.is_fused:
                    if self._hook_fused_container(idx, layer_name, container):
                        accepted.append(container)
                        fused_containers += 1
                    continue
                count = self._hook_expert_modules(idx, container, packed_bytes)
                if count:
                    accepted.append(container)
                    hooked_experts += count

            if not accepted:
                continue
            # Recomputed rather than taken from the layout, because a container detection was happy
            # with may still have been rejected just above; its tensors have to go back to the
            # layer's own stream or nothing would ever load them.
            owned = {key for container in accepted for key in container.keys}
            self._non_expert_keys[idx] = [key for key in shapes if key not in owned]
            self._expert_layouts[idx] = ExpertLayout(containers=tuple(accepted),
                                                     other_keys=tuple(self._non_expert_keys[idx]))

        self._expert_streaming = bool(hooked_experts or fused_containers)
        self._report_expert_streaming(hooked_experts, fused_containers, skipped)

    def _layer_tensor_specs(self, layer_name):
        """Every tensor in a layer's shard, from the header alone -- shapes and packed sizes.

        Detection needs the shapes; the cache needs the byte counts to size an expert before it
        decides whether to keep it. Neither reads a byte of tensor data.
        """
        placements = self.loader.plan(layer_name).placements
        return ({p.name: p.shape for p in placements},
                {p.name: p.nbytes for p in placements})

    def _cannot_stream_alone(self, container):
        """Why a container has to stream with its layer, or None when it can stream by itself."""
        if container.is_fused:
            for key in container.keys:
                if self.quant.needs_quantizer(key):
                    return ("the checkpoint's quantizer reconstructs this tensor and needs all of "
                            "it, not the routed rows")
        return None

    def _configured_containers(self, shapes):
        """The container a subclass named outright, for when structure alone did not find it.

        `expert_prefix` used to be the only way in. It stays as the manual override that this
        project's working rules ask of every automatic decision -- measurement first, an override
        behind it -- and it runs only when detection came back empty, so a checkpoint whose experts
        are found structurally is never second-guessed by a stale hint.
        """
        prefix = self.layer_names_dict.get('expert_prefix')
        if not prefix:
            return []
        marker = f'.{prefix}.'
        per_expert = {}
        for key in shapes:
            pos = key.find(marker)
            if pos == -1:
                continue
            head = key[pos + len(marker):].split('.', 1)[0]
            if head.isdigit():
                per_expert.setdefault(int(head), []).append(key)
        if len(per_expert) < 2:
            return []
        return [ExpertContainer(layout=LAYOUT_MODULE_LIST, path=prefix,
                                num_experts=len(per_expert),
                                top_k=resolve_top_k(self.config, len(per_expert)),
                                expert_keys={i: tuple(k) for i, k in per_expert.items()})]

    def _hook_expert_modules(self, idx, container, packed_bytes):
        """A forward hook per expert module. Returns how many were hooked.

        Each expert becomes a cache entry in its own right, keyed the way the cache already expects
        one, so residency and eviction are decided for it individually rather than for the layer it
        happens to sit in.
        """
        try:
            experts = self.layers[idx].get_submodule(container.path)
        except AttributeError:
            return 0
        hooked = 0
        for expert_idx, keys in container.expert_keys.items():
            try:
                module = experts.get_submodule(str(expert_idx))
            except AttributeError:
                continue
            key = (idx, expert_kind(expert_idx))
            self._expert_tensor_keys[key] = keys
            self._expert_byte_counts[key] = sum(packed_bytes.get(name, 0) for name in keys)
            module._rocketllm_expert = key
            module.register_forward_pre_hook(self._expert_pre_hook)
            module.register_forward_hook(self._expert_post_hook)
            hooked += 1
        return hooked

    def _hook_fused_container(self, idx, layer_name, container):
        """Watch the router, then stream the rows it chose into the fused tensor."""
        layer = self.layers[idx]
        try:
            experts = layer.get_submodule(container.path)
            router = layer.get_submodule(container.router_path)
        except AttributeError:
            return False

        selection = RouterSelection(container.num_experts, container.top_k,
                                    path=f"{layer_name}.{container.path}")
        router._rocketllm_selection = selection
        router.register_forward_hook(self._router_hook)

        experts._rocketllm_fused = (idx, container, selection)
        experts.register_forward_pre_hook(self._fused_pre_hook)
        experts.register_forward_hook(self._fused_post_hook)
        return True

    def _report_expert_streaming(self, hooked_experts, fused_containers, skipped):
        """Say what was found, at load, once. A user who lost the fast path has to be able to see."""
        for line in summarize_experts(self._expert_layouts):
            print(f"MoE: {line}")
        if hooked_experts:
            print(f"per-expert streaming: {hooked_experts} expert modules across "
                  f"{len(self._expert_layouts)} layers are cached individually, so residency is "
                  f"decided per expert rather than per layer.")
        if fused_containers:
            print(f"fused-expert streaming: {fused_containers} containers read only the rows their "
                  f"router selects, so a token costs its own experts' bytes, not the layer's.")
        for reason in dict.fromkeys(reason for _, reason in skipped):
            print(f"expert container streamed with its layer instead: {reason}")

    def _expert_pre_hook(self, module, args):
        """Hand an expert to the cache, exactly as a dense module's hook does.

        Going through the cache rather than reading the shard here is what stops a mixture paying
        for the same expert on every token. An expert that is still resident from the last token
        costs nothing; one that is not is read, and then competes for residency on its measured
        popularity -- the LFU half of the cache's policy, which exists for precisely this and had no
        caller until now.
        """
        self.cache.acquire(module._rocketllm_expert)

    def _expert_post_hook(self, module, args, output):
        self.cache.release(module._rocketllm_expert)
        return output

    def _router_hook(self, module, args, output):
        module._rocketllm_selection.observe(output)
        return output

    def _fused_pre_hook(self, module, args):
        idx, container, selection = module._rocketllm_fused
        layer_name = self.layer_names[idx]
        rows = selection.take()
        if rows is None:
            # The router's choice could not be read. Every row is always correct, and costs what a
            # whole layer has always cost -- which is the behaviour this replaces, not a regression.
            state_dict = load_layer_subset(self.checkpoint_path, layer_name, container.keys)
            module._rocketllm_moved = self.move_layer_to_device(state_dict)
            return
        compact = load_layer_rows(self.checkpoint_path, layer_name,
                                  {key: rows for key in container.keys})
        module._rocketllm_moved = self._place_expert_rows(container, rows, compact)

    def _fused_post_hook(self, module, args, output):
        for param_name in getattr(module, '_rocketllm_moved', []):
            set_module_tensor_to_device(self.model, param_name, 'meta')
        module._rocketllm_moved = []
        return output

    def _place_expert_rows(self, container, rows, compact):
        """Bind a fused expert tensor holding only the rows this token routed to.

        The parameter keeps its full width. The module's forward indexes it by expert ordinal, and
        narrowing it would mean rewriting a forward that differs per architecture -- precisely what
        this engine exists not to do, and what stops it needing a code change per model. So the
        destination is allocated at the checkpoint's shape and the routed rows are scattered in.

        Zero is the right filler for the rest, and not by luck: a row the router did not choose is
        multiplied by a zero routing weight, which is what routing *is*. Architectures put the zero
        in different places -- Llama 4 scales the expert's input by the score, others mask its
        output or never call it -- and every one of them leaves the unrouted weights unable to reach
        the result. What this saves is the read. Storage and the link carry the token's own experts
        rather than the whole layer's, which is the tier that dominates on a streaming machine.
        Device memory is unchanged, because the full-width tensor still has to exist.
        """
        moved = []
        index = torch.tensor(list(rows), device=self.running_device, dtype=torch.long)
        for key, value in compact.items():
            target = self.quant.target_dtype(key, value)
            full = torch.zeros(container.fused_shapes[key], dtype=target,
                               device=self.running_device)
            self._adopt_checkpoint_shape(key, full)
            full.index_copy_(0, index, value.to(device=self.running_device, dtype=target))
            set_module_tensor_to_device(self.model, key, self.running_device, value=full)
            moved.append(key)
        return moved

    def _knob(self, name, fallback=0):
        """A tuning value: the caller's override, else the profile's measurement, else a floor."""
        override = self._overrides.get(name)
        if override is not None:
            return int(override)
        if self.profile is not None:
            derivation = self.profile.derived.get(name)
            if derivation is not None:
                return int(derivation.value)
        return int(fallback)

    def _build_cache(self):
        """Hand residency over to the cache.

        Everything the cache needs to size itself comes from the HardwareProfile, and every one of
        those numbers was measured on this machine rather than chosen. The window is the memory
        answer -- how many of the largest layer fit in the window budget -- clamped to at least one,
        because a cache that cannot hold a single layer cannot run a forward at all.
        """
        largest = max((self._layer_packed_bytes(i) for i in self._streamed_indices), default=0)
        window_budget = self._knob("window_budget_bytes")
        window = max(1, window_budget // largest) if largest else 1
        ceiling = self._overrides.get("window_max")
        if ceiling is None and self.profile is not None:
            ceiling = self.profile.window_max(largest)
        if ceiling:
            window = max(1, min(window, int(ceiling)))

        host_bytes = self._knob("host_cache_bytes")
        self._keep_host_copies = host_bytes > 0

        device_bytes = self._knob("usable_device_bytes")
        if self._overrides.get("reserve_bytes") is not None and self.profile is not None:
            # usable_device_bytes was derived from the measured reserve, so an overridden reserve
            # has to be folded back in or the override would silently do nothing.
            total = self.profile.device_total_bytes or 0
            device_bytes = max(0, total - int(self._overrides["reserve_bytes"]))
        pinned = self._plan_pins(device_bytes, window_budget, largest)

        self.cache = TieredWeightCache(
            fetch=self._cache_fetch, sizer=self._packed_bytes,
            to_device=self._cache_to_device, to_host=self._cache_to_host,
            discard=self._cache_discard,
            sequence=self._cache_sequence if self.prefetching else None,
            device_bytes=device_bytes, host_bytes=host_bytes, window=window, pinned=pinned,
            profile=self.profile, prefetch_workers=self._knob("io_workers", 1))

        print(f"cache: window {window} layers, device budget "
              f"{device_bytes / 1024 ** 3:.1f}GB, host tier {host_bytes / 1024 ** 3:.1f}GB, "
              f"{len(pinned)} pinned, pin_policy={self.pin_policy}")

    def _plan_pins(self, device_bytes, window_budget, largest):
        """Which modules to keep resident for the whole run.

        Every streamed module here is touched once per token, so they all sit in the same priority
        class and the ranking reduces to size: pinning the smallest first keeps the most of them,
        which is what buys the hit rate. An MoE's experts are ranked separately by the placement
        module and are not part of this list.
        """
        if self.pin_policy == "off":
            return ()
        budget = pin_budget_from(device_bytes, window_budget)
        candidates = [PinCandidate(key=(idx, KIND_DENSE),
                                   packed_bytes=self._layer_packed_bytes(idx),
                                   priority=CLASS_ALWAYS, accesses_per_token=1.0)
                      for idx in self._streamed_indices]
        return plan_pins(candidates, budget).pinned

    def _load_streamed_layer(self, idx):
        """Load one streamed module's weights. Experts are excluded when they stream themselves."""
        keys = self._non_expert_keys.get(idx) if getattr(self, '_expert_streaming', False) else None
        if keys is None:
            return self.load_layer_to_cpu(self.layer_names[idx])
        return load_layer_subset(self.checkpoint_path, self.layer_names[idx], keys)

    # ---- the cache's view of a streamed module ----------------------------------------------

    def _cache_fetch(self, key):
        """Storage tier: read a unit's tensors. Runs on a prefetch worker, so it only reads."""
        if is_expert(key[1]):
            return _ResidentModule(key, host=load_layer_subset(
                self.checkpoint_path, self.layer_names[key[0]], self._expert_tensor_keys[key]))
        return _ResidentModule(key, host=self._load_streamed_layer(key[0]))

    def _cache_to_device(self, resident):
        """Device tier: place the weights and bind them to the module.

        Always on the thread that called acquire. Binding parameters mutates the model, and doing
        that from a prefetch worker while a forward is running is a race with no recovery.
        """
        if resident.moved is None:
            resident.moved = self.move_layer_to_device(resident.host)
            # The host copy is only worth its RAM if this entry can ever be evicted to the host
            # tier. A pinned entry never is, and neither is anything when there is no host tier, so
            # in both cases holding the CPU-side copy is pure waste -- and on a model that fits
            # entirely on the device that is every layer.
            pinned = resident.key in self.cache.pinned
            if not self._keep_host_copies or pinned:
                resident.host = None
        return resident

    def _cache_to_host(self, resident):
        """Host tier: unbind from the module but keep the CPU copy to serve the next hit."""
        self._unbind(resident)
        return resident

    def _cache_discard(self, resident):
        """The entry is leaving the cache: unbind and let the host copy go."""
        if resident is None:
            return
        self._unbind(resident)
        resident.host = None

    def _unbind(self, resident):
        """Send this module's parameters back to meta.

        Sending the weights to meta is all the releasing this needs to do. It used to also empty the
        allocator cache, which handed the blocks back to the driver and made the next layer pay for
        a fresh, synchronizing allocation -- every layer, every token. The expensive release happens
        once per generation, in reset().
        """
        for param_name in resident.moved or ():
            set_module_tensor_to_device(self.model, param_name, 'meta')
        resident.moved = None

    def _cache_sequence(self, key, width):
        """The modules that will be wanted after `key`, for the cache to read ahead.

        A forward runs embed -> layers -> norm -> lm_head in order, so lookahead is just the next
        few streamed indices. There is no cross-layer lookahead for MoE experts and none is implied
        here: layer L's router has not run when layer L is being fetched, so which experts layer
        L+1 will want is not merely unknown, it is undefined. The guard below states that rather
        than leaving it to the fact that nothing currently asks.
        """
        if is_expert(key[1]):
            return []
        idx = key[0]
        upcoming = []
        for step in range(1, max(1, width) + 1):
            nxt = idx + step
            if nxt not in self._streamed_set:
                break
            upcoming.append((nxt, KIND_DENSE))
        return upcoming

    def _packed_bytes(self, key):
        """Packed bytes of one cached unit, dense module or single expert."""
        if is_expert(key[1]):
            return self._expert_byte_counts.get(key, 0)
        return self._layer_packed_bytes(key[0])

    def _layer_packed_bytes(self, idx):
        """Packed bytes of one streamed module, read from the shard header rather than the data."""
        cached = self._layer_bytes.get(idx)
        if cached is not None:
            return cached
        size = 0
        try:
            layout = self.loader.plan(self.layer_names[idx],
                                      keys=self._non_expert_keys.get(idx)
                                      if getattr(self, '_expert_streaming', False) else None)
            size = layout.total_bytes
        except Exception:  # noqa: BLE001 - sizing must not be able to fail a load
            try:
                size = os.path.getsize(self.loader.shard_path(self.layer_names[idx]))
            except OSError:
                size = 0
        self._layer_bytes[idx] = size
        return size

    def _pre_hook(self, module, args):
        key = (module._rocketllm_idx, KIND_DENSE)
        self.cache.acquire(key)
        self.cache.prefetch_window(key)

    def _post_hook(self, module, args, output):
        self.cache.release((module._rocketllm_idx, KIND_DENSE))
        return output

    # ---- lifecycle --------------------------------------------------------------------------

    def _staging_budget_bytes(self):
        """How much host memory the staging pool may hold, from the hardware profile."""
        try:
            self.profile = HardwareProfile.load_or_probe(weights_path=self.checkpoint_path,
                                                         device=self.running_device)
            return self.profile.derived["staging_pool_bytes"].value
        except Exception as exc:  # noqa: BLE001 - loading must not fail over a tuning knob
            self.profile = None
            caps.announce_once(
                "profile-unavailable",
                f"could not build a hardware profile ({exc}); the staging pool runs unpooled, "
                f"which is slower but produces the same result.", logging.INFO)
            return 0

    def reset(self):
        """Release cached device blocks and host garbage. Between generations, never between them.

        A streaming run touches every layer on every token, so anything done per layer is done
        hundreds of times per token. Releasing memory is only worth paying for when something else
        might want it, which is when a generation ends.

        Staging buffers deliberately survive: they are what makes the next generation cheap, and
        holding them is the whole point of the pool. close() is what gives them back.
        """
        # Drop the cache's residency first: it is holding device buffers, and freeing them before
        # the allocator is asked to hand memory back is the whole point of doing this here.
        cache = getattr(self, "cache", None)
        if cache is not None:
            cache.clear()
        self._end_generation()

    def _end_generation(self):
        """Settle everything in flight and hand freed blocks back, keeping cache residency.

        What a generation boundary actually requires: no copy still reading a staging buffer, and
        the allocator's freed blocks returned. Evicting the weights themselves is a separate
        decision, and usually the wrong one -- see generate().
        """
        # Nothing may be in flight when the allocator is asked to hand memory back, and a staged
        # buffer cannot be freed while a copy is still reading it.
        self.transfer.drain()
        clean_memory(self.running_device)

    def close(self):
        """Shut the model down for good: stop the prefetch worker and release everything."""
        cache = getattr(self, "cache", None)
        if cache is not None:
            cache.close()
        loader = getattr(self, "loader", None)
        if loader is not None:
            loader.close()
        self.prefetching = False
        self.transfer.close()
        self.staging_pool.clear()
        self.reset()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ---- delegation to the underlying transformers model ------------------------------------

    def generate(self, *args, **kwargs):
        try:
            return self.model.generate(*args, **kwargs)
        finally:
            # One release per generation, in place of one per layer per token -- but NOT a cache
            # clear. Residency is the whole point of the cache, and a second generation that had to
            # re-read every layer from storage would pay the full first-token cost again. The cache
            # is bounded by the device budget, so keeping it costs nothing that was not already
            # budgeted. Call reset() explicitly to give it back.
            self._end_generation()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
