
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
import time

import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.generation import GenerationMixin
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer

from . import conversion as checkpoint_conversion
from .hw import caps
from .hw.caps import get_caps
from .hw.profile import HardwareProfile
from .quant import detect_backend
from .quant.safetensors_quant import announce_backend
from .memory import (CLASS_ALWAYS, KIND_DENSE, PinCandidate, TieredWeightCache, VramBudget,
                     configure_allocator, expert_kind, is_expert, pin_budget_from, plan_pins)
from .quant.kv_cache import KV_CHOICES, KV_FP16, KVCacheConfig, build_kv_cache, resolve_kv_cache
from .moe import (LAYOUT_MODULE_LIST, ExpertContainer, ExpertLayout, ExpertResidency,
                  RouterSelection, detect_expert_layout, resolve_top_k, shared_kind,
                  summarize as summarize_experts)
from .spec import (SPEC_CHOICES, DraftModel, SamplingParams, SpeculativeDecoder,
                   announce as announce_speculation, lookahead_ceiling, resolve_speculation)
from .streaming import HostStagingPool, LayerLoader, WeightTransfer
from .profiler import LayeredProfiler

from .utils import clean_memory, load_layer, load_layer_rows, load_layer_subset, \
    find_or_create_local_splitted_path, load_checkpoint_config, reject_compression_argument
from .persist import ModelPersister


#: Auto classes tried, in order, to build the empty skeleton.
#:
#: A causal LM first, because that is what nearly every checkpoint is and what a remote-code model
#: registers itself as. Then the two multimodal factories: a vision-language config is registered
#: only there, and ``AutoModelForCausalLM`` refuses it outright with a ValueError listing every
#: config class it does know -- which is the failure this list exists to get past.
#:
#: Named rather than imported, because which of them exists differs across the transformers range
#: this project supports. One that is absent is skipped, not an ImportError at module load.
MODEL_FACTORIES = ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForVision2Seq")

#: Keys in a preprocessor config that only an image processor writes. A checkpoint that ships one
#: has a vision tower whatever it calls itself, and a speech checkpoint's preprocessor has none of
#: them -- which keeps this from claiming a processor for a model that has no images to process.
_IMAGE_PREPROCESSOR_KEYS = ("image_processor_type", "image_mean", "image_std", "crop_size")

#: Substrings that mark a generate() keyword as a non-text model input. Speculation reads token ids
#: and nothing else, so a call carrying any of these has to take the ordinary path or the draft
#: would propose from a prompt with the image removed -- fluent, plausible, and about the wrong
#: picture. No text-generation argument contains any of these.
_NON_TEXT_INPUT_MARKERS = ("pixel", "image", "video", "audio", "vision", "input_features")

_NO_MODEL_FACTORY = """\
RocketLLM could not build a model for this checkpoint. transformers has its config class \
({config}), but none of its model factories accepts one:

{attempts}

The skeleton is built through transformers' auto classes on purpose -- it is what lets a newly \
released architecture stream here without a code change -- so a config no factory accepts is \
usually one whose model class lives in a newer transformers than the {version} installed, or one \
for a task this engine does not serve. RocketLLM generates text; an encoder-only, embedding or \
speech checkpoint has no path through it."""


def _brief(error, limit=200):
    """An exception's first line, capped. Some of these run to two kilobytes of config classes."""
    text = str(error).strip().splitlines()
    first = text[0] if text else type(error).__name__
    return first if len(first) <= limit else first[:limit - 3] + "..."


def _refuses_config(error):
    """Whether an auto class rejected the config outright rather than the attention choice.

    Read off transformers' own message, which is the only place the distinction is recorded. Being
    wrong about it costs one extra build attempt and a differently-worded log line, never a
    different outcome, so a message that gets reworded upstream degrades rather than breaks.
    """
    text = str(error)
    return "Unrecognized configuration class" in text or "for this kind of AutoModel" in text


def _model_factories():
    """The auto classes this transformers actually has, in the order they should be tried."""
    import transformers

    found = []
    for name in MODEL_FACTORIES:
        factory = getattr(transformers, name, None)
        if factory is not None:
            found.append((name, factory))
    return found


def checkpoint_dtype(config):
    """The dtype the checkpoint says its weights are in, wherever it says it.

    Three spellings have to be read. ``torch_dtype`` up to transformers 4.55, ``dtype`` from 4.56,
    and -- the one that matters here -- on a multimodal config neither may be set at the top at all,
    because the weights' dtype is a property of each sub-model. Qwen3.5 declares it only on
    ``text_config``.

    Reading only the outer name silently produced None, and None fell through to float16. That is
    the one default this project cannot afford to get wrong by accident: fp16's range overflows on a
    deep model and corrupts the output without raising, and these are 40- and 64-layer models. So
    the search descends, and returns None rather than guessing when nothing says.
    """
    from transformers import PretrainedConfig

    def look(cfg, depth=0):
        if depth > 2:
            return None
        for attribute in ("dtype", "torch_dtype"):
            value = getattr(cfg, attribute, None)
            if isinstance(value, str):
                value = getattr(torch, value, None)
            if isinstance(value, torch.dtype):
                return value
        for sub in vars(cfg).values():
            if isinstance(sub, PretrainedConfig):
                found = look(sub, depth + 1)
                if found is not None:
                    return found
        return None

    return look(config)


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

    def __init__(self, model_local_path_or_repo_id, device=None, dtype=None, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 vram_reserve=None, host_cache_gb=None, io_workers=None, window_max=None,
                 shard_handle_limit="auto", pin_policy="auto", expert_residency="auto",
                 kv_cache="auto", draft_model=None, speculative="auto"):
        """
        Parameters
        ----------
        model_local_path_or_repo_id : str or Path
            path to the local model checkpoint or huggingface repo id
        device : str, optional
            backend to run on, e.g. "cuda:0", "mps", "cpu". Default: whatever this machine
            actually has, queried at startup like every other hardware fact here. It used to
            default to "cuda:0", which meant a machine without an NVIDIA card crashed in the
            constructor rather than running slower on the backend it does have.
        dtype : torch.dtype, optional
            runtime dtype; defaults to the model's own config.torch_dtype (usually bfloat16 for
            modern models). float16 has too narrow a range for very deep models and overflows to
            inf/NaN, which silently corrupts the output, so we don't force it.
        max_seq_len : int, optional
            sequence length the streaming engine is set up for, by default 512. A property of the
            model and the workload rather than of the machine: how much context actually fits is
            the KV cache's decision, made against the measured device budget.
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
        shard_handle_limit: str or int, optional
            how many shard files may stay memory-mapped at once. Mapping is the cheaper read --
            safetensors hands back tensors that alias the file rather than copies of it -- but on
            an operating system that charges mappings against a commit limit they stay charged for
            as long as any tensor read from them is alive. "auto" (default) compares this
            checkpoint against the commit headroom measured at load and maps where it fits, reading
            byte ranges where it does not. "unbounded" always maps, "direct" never does, and an
            integer maps with at most that many shards held. Both routes return the same bytes, so
            this cannot change a single token.
        pin_policy: str, optional
            "auto" (default) ranks candidates by access-frequency-per-packed-byte and fills the pin
            budget; "off" pins nothing and streams everything, which is the pure-streaming
            configuration and is what a device with no spare memory gets anyway.
        expert_residency: str, optional
            "auto" (default) counts how often each expert is routed to and keeps the popular ones
            resident; "off" leaves a mixture's experts to the ordinary cache without ever pinning
            one. Exists so the policy can be measured against its own absence on the same build,
            and because a model that routes uniformly has nothing for it to exploit.
        kv_cache: str, optional
            How to hold the KV cache. "auto" (default) keeps it in the compute dtype when the
            weights fit resident with headroom to spare, and quantizes to int4 when they do not --
            because only in the second case is a byte of context a byte of weights re-read next
            token. "fp16" and "int4" force the choice. "hqq" and "quanto" delegate to transformers'
            own quantized cache on those backends, as a reference to check the int4 path against;
            both are optional packages and fall back with a message when absent.

            This is INDEPENDENT of the weight format and must stay so: the checkpoint's
            quantization describes what someone else produced, this describes what fits here.
        draft_model: str or Path, optional
            a small model, from the same family and sharing the target's tokenizer, kept fully
            resident to propose tokens that one pass of this model then verifies. It must share the
            vocabulary and tokenizer exactly; that is checked at load and refused with a message
            naming what differs, because a mismatch produces plausible wrong output rather than an
            error. Nothing happens without one.
        speculative: str, optional
            "auto" (default) takes the profile's measured recommendation: speculation pays when a
            streaming pass costs far more than device memory does, and costs residency when it does
            not, so on a machine where the model already fits it is left off. "on" and "off" force
            it. "auto" never enables it silently against the profile's own measurement, and it never
            enables it on a machine whose bandwidths could not be measured at all.
        """

        # First thing, before anything in this constructor touches the device. The allocator reads
        # its configuration once, when the context is created, so asking for expandable segments
        # after the hardware probe has run is asking too late -- and under a streaming workload,
        # where a layer's blocks are carved and released hundreds of times per token, that setting
        # is the difference between reusing those blocks and stranding them.
        self.allocator = configure_allocator(device)

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

        # Queried, not assumed. `device=None` means "the fastest backend this machine actually
        # has", resolved the same way every other hardware fact here is, so constructing a model
        # on a box with no accelerator is a slower run rather than a crash.
        self.device = caps.resolve_device(device)
        self.running_device = str(self.device)

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
            # load_checkpoint_config on the second attempt, not the first: an architecture this
            # transformers does not have fails BOTH ways, and only the last failure should be the
            # one that gets explained.
            self.config = load_checkpoint_config(
                self.model_local_path, trust_remote_code=True, hf_token=hf_token)
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
            requested = checkpoint_dtype(self.config)
            if requested is None:
                # Nothing in the checkpoint says. Ask for the widest of the two candidates and let
                # the device decide: on hardware with bf16 that is what runs, and on hardware
                # without it select_compute_dtype degrades to fp16 and says so out loud. Defaulting
                # straight to fp16 here, as this used to, skipped that warning entirely.
                requested = torch.bfloat16
                caps.announce_once(
                    "dtype-undeclared",
                    "this checkpoint does not declare the dtype its weights are stored in, so the "
                    "run uses the widest dtype this device supports. Pass dtype= to choose.",
                    logging.INFO)
            dtype = self.caps.select_compute_dtype(requested)
        self.running_dtype = dtype
        self.dtype = self.running_dtype
        self._declare_runtime_dtype()

        self.generation_config = self.get_generation_config()
        #: The checkpoint's multimodal processor, or None for a text-only model. The tokenizer is
        #: taken from inside it where it has one, so the two cannot disagree about the vocabulary.
        self.processor = self.init_processor(hf_token=hf_token)
        self.tokenizer = (getattr(self.processor, "tokenizer", None)
                          or self.get_tokenizer(hf_token=hf_token))

        self.prefetching = prefetching
        #: Module path -> whether the model class has a home for it. Memoised because
        #: the question is asked of every tensor of every layer on every pass.
        self._homes = {}
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
        if expert_residency not in ("auto", "off"):
            raise ValueError(
                f"expert_residency must be 'auto' or 'off', not {expert_residency!r}")
        self.expert_residency = expert_residency
        if kv_cache not in KV_CHOICES:
            raise ValueError(f"kv_cache must be one of {', '.join(KV_CHOICES)}, not {kv_cache!r}")
        self.kv_cache_setting = kv_cache
        self.kv_cache_choice = None
        if speculative not in SPEC_CHOICES:
            raise ValueError(
                f"speculative must be one of {', '.join(SPEC_CHOICES)}, not {speculative!r}")
        self.speculative_setting = speculative
        self.draft_path = draft_model
        self.draft = None
        self.spec = None
        self._layer_bytes = {}

        # Staging buffers are sized from the machine, not from a constant. Probing is cached by
        # hardware fingerprint, so this is a one-off cost; if it fails for any reason the pool
        # runs with a zero budget, which means no pooling and no pinning -- slower, still correct.
        self.staging_pool = HostStagingPool(self.caps, self._staging_budget_bytes())

        # Built before anything allocates on the device, because it is what asks for expandable
        # segments and that setting is read once, when the context is created. It measures from
        # here on; the cache is wired to it in _build_cache, once there is a cache to resize.
        # Configured at the top of the constructor already, so it is not attempted a second time
        # here -- doing so would report "too late" against a context this object created itself.
        self.budget = VramBudget(device_caps=self.caps, profile=self.profile,
                                 reserve_bytes=self._overrides.get("reserve_bytes"),
                                 configure_allocator_env=False)
        # The copy stream, and the thing that owns a staged buffer's lifetime until its event
        # fires. Built here because the pool has to know how to ask it for finished transfers
        # before it decides it has no free buffer.
        self.transfer = WeightTransfer(self.caps, pool=self.staging_pool)
        # Header-only reads for sizing, and the byte-range reads the cache's storage tier uses.
        self.loader = LayerLoader(self.checkpoint_path, self.staging_pool, profile=self.profile,
                                  io_workers=self._overrides.get("io_workers"),
                                  shard_handle_limit=shard_handle_limit)

        self.init_model()

        # compute layer count from the instantiated model
        layers_count = len(self.submodule(self.layer_names_dict["layer_prefix"]))

        self.layer_names = [self.layer_names_dict['embed']] + \
                           [f'{self.layer_names_dict["layer_prefix"]}.{i}' for i in range(layers_count)] + \
                           [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']]

        self.max_seq_len = max_seq_len

        self.set_layers_from_layer_names()
        self._load_resident_modules()
        self._install_streaming_hooks()
        self._build_cache()
        # Last, because the draft takes device memory away from the cache that was just sized, and
        # the budget has to be told so the cache gives those bytes back rather than discovering
        # them missing on the first token.
        self._setup_speculation()

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

    def declares_vision_components(self):
        """Whether this checkpoint carries a vision tower, from what it says and what it ships.

        Structural, like every other decision here: a nested vision sub-config, or the image
        preprocessor a checkpoint has to ship for one. Never an architecture name, so a
        vision-language model released next month is recognised without a code change.
        """
        for name, value in vars(self.config).items():
            if name.endswith("vision_config") and value is not None:
                return True
        for name in ("preprocessor_config.json", "processor_config.json"):
            path = Path(self.model_local_path) / name
            if not path.exists():
                continue
            try:
                with open(path, "rb") as handle:
                    settings = json.load(handle)
            except Exception:  # noqa: BLE001 - an unreadable hint is not a vision tower
                continue
            if isinstance(settings, dict) and any(key in settings
                                                  for key in _IMAGE_PREPROCESSOR_KEYS):
                return True
        return False

    def init_processor(self, hf_token=None):
        """The checkpoint's own multimodal processor, or None for a text-only model.

        A processor is what turns an image into the tensors a vision-language model's forward reads,
        and it is the only thing that knows how many placeholder tokens one image expands to for
        THIS checkpoint -- a count that depends on the image's size against the model's patch and
        merge sizes, and that no amount of care with the tokenizer alone reproduces.

        It deliberately does not REPLACE the tokenizer. Every text path in this engine calls the
        tokenizer directly, and a processor's ``__call__`` takes images as its first argument, so
        substituting one would silently pass prompts in as pictures. It is carried alongside instead.
        """
        if not self.declares_vision_components():
            return None
        try:
            from transformers import AutoProcessor
        except ImportError:  # pragma: no cover - AutoProcessor predates the supported range
            return None
        kwargs = {'trust_remote_code': True}
        if hf_token is not None:
            kwargs['token'] = hf_token
        try:
            return AutoProcessor.from_pretrained(self.model_local_path, **kwargs)
        except Exception as exc:  # noqa: BLE001 - text generation must survive this
            caps.announce_once(
                "processor-unavailable",
                f"this checkpoint declares vision components but its processor could not be built: "
                f"{_brief(exc, limit=400)}. Text generation is unaffected; image inputs will be "
                f"refused. A missing imaging package is the usual cause and both live in one "
                f"extra -- `pip install \"rocketllm[vision]\"` -- but note that torchvision has to "
                f"match your torch build, so on CUDA install it from the index torch came from.",
                logging.WARNING)
            return None

    # ---- model construction -----------------------------------------------------------------

    def _declare_runtime_dtype(self):
        """Tell the config what dtype this run is actually in, before the model is built.

        The model is built from the config, so the config's dtype is what every meta placeholder
        gets -- and a placeholder's dtype is not inert. accelerate's placement casts an incoming
        value to the existing parameter's dtype whenever it is not told otherwise, so a config
        saying bf16 silently pulls every weight back to bf16 however carefully the runtime dtype was
        chosen. With `dtype=` left alone the two agree and nothing is visible; override it and the
        model declares one dtype while the engine intends another, which surfaces as a dtype
        mismatch inside the first matmul rather than as anything pointing back here.

        Nested sub-configs are set too, for the same reason the attention implementation is: a
        multimodal wrapper keeps the real decoder under `text_config`, and a value set only on the
        outer config never reaches it.
        """
        from transformers import PretrainedConfig

        def declare(cfg, depth=0):
            if depth > 2:
                return
            # `torch_dtype` up to transformers 4.5x, `dtype` from 4.56 on. Set whichever the object
            # already carries so this works either side of the rename, and `torch_dtype` regardless,
            # since that is what the versions this project supports read.
            cfg.torch_dtype = self.running_dtype
            if hasattr(cfg, "dtype"):
                cfg.dtype = self.running_dtype
            for sub in vars(cfg).values():
                if isinstance(sub, PretrainedConfig):
                    declare(sub, depth + 1)

        declare(self.config)

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

    def _build_empty_model(self, factory):
        """One auto class's attempt at the skeleton on meta, or ``(None, why it refused)``.

        sdpa first, then eager: some (often remote-code) architectures do not implement sdpa and
        also default to it, so eager has to be asked for explicitly or transformers re-selects sdpa
        and errors again. A factory with no entry for this config refuses both ways, which is a
        different thing entirely -- that is the caller's cue to try the next factory rather than to
        report an attention problem.
        """
        last = None
        for impl in ("sdpa", "eager"):
            try:
                self._propagate_attn_implementation(impl)
                with init_empty_weights(include_buffers=False):
                    return factory.from_config(self.config, attn_implementation=impl,
                                               trust_remote_code=self.trust_remote_code), None
            except (ValueError, TypeError) as exc:
                last = exc
                if _refuses_config(exc):
                    break
                if impl == "sdpa":
                    print(f"attn_implementation='sdpa' not available ({_brief(exc)}), "
                          f"falling back to eager attention")
        return None, last

    def init_model(self):
        # Build the real model on meta (no memory). include_buffers=False so non-persistent
        # buffers such as rotary inv_freq are actually computed (they aren't in the checkpoint).
        #
        # Several factories are tried because a vision-language config is registered under a
        # different one: AutoModelForCausalLM rejects Qwen2.5-VL's config outright, and did so with
        # a ValueError so long that it read like an attention-implementation problem. Which factory
        # succeeded is a property of the checkpoint, not of the machine, so it is discovered rather
        # than configured.
        # Which class to build is the CHECKPOINT's statement, not the first factory's guess. One
        # config can be registered under several auto classes that build different models from it:
        # Qwen3.5's config is registered under AutoModelForCausalLM, which builds the text-only
        # Qwen3_5ForCausalLM -- no vision tower, and the checkpoint's `visual` weights silently
        # never loaded. So every factory that accepts the config is asked what it would build, and
        # the one that matches `architectures[0]` wins.
        declared = next(iter(getattr(self.config, "architectures", None) or ()), None)

        self.model = None
        attempts = []
        fallback = None
        for name, factory in _model_factories():
            model, refusal = self._build_empty_model(factory)
            if model is None:
                attempts.append(f"  {name}: {_brief(refusal)}")
                continue
            built = type(model).__name__
            if declared is None or built == declared:
                self.model, self.model_factory = model, name
                break
            # It builds something, but not what the checkpoint says it is. Keep it only as a last
            # resort and go on looking.
            if fallback is None:
                fallback = (name, model, built)
                attempts.append(f"  {name}: builds {built}, not the declared {declared}")

        if self.model is None and fallback is not None:
            name, model, built = fallback
            self.model, self.model_factory = model, name
            caps.announce_once(
                "model-class-mismatch",
                f"this checkpoint declares {declared}, which no auto class in this transformers "
                f"builds; {name} was used instead and produced {built}. Anything the declared class "
                f"has and this one does not -- a vision tower, an extra head -- is not part of the "
                f"model that will run.", logging.WARNING)

        if self.model is None:
            raise ValueError(_NO_MODEL_FACTORY.format(
                config=type(self.config).__name__, attempts="\n".join(attempts) or "  (none)",
                version=__import__("transformers").__version__))

        if attempts:
            print(f"built {type(self.model).__name__} with {self.model_factory}; "
                  f"{len(attempts)} other factory attempt(s) did not fit this checkpoint.")

        #: What this model class says about reading its own checkpoints: the renames between
        #: checkpoint tensor names and parameter names, and the expert fusions where the two
        #: disagree about shape rather than about spelling. Read from transformers rather than
        #: restated here -- see rocketllm/conversion.py.
        self.conversion = checkpoint_conversion.describe(self.model)
        self._to_module_name = self.conversion.rename if self.conversion.renames else None

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

    # ---- checkpoint names vs module names ---------------------------------------------------

    def module_name(self, checkpoint_name):
        """Where the model keeps what the checkpoint calls ``checkpoint_name``.

        The identity for every ordinary checkpoint. See :func:`checkpoint_name_mapper` for the case
        it exists to serve, and note that it works on module paths as well as parameter names --
        the rules are anchored prefixes, so ``model.layers`` renames exactly as
        ``model.layers.0.mlp.up_proj.weight`` does.
        """
        translate = getattr(self, "_to_module_name", None)
        return checkpoint_name if translate is None else translate(checkpoint_name)

    def _to_module_namespace(self, state_dict):
        """Re-key a shard from the checkpoint's names to the model's, where the two differ."""
        translate = getattr(self, "_to_module_name", None)
        if translate is None:
            return state_dict
        renamed = OrderedDict()
        for name, value in state_dict.items():
            renamed[translate(name)] = value
        return renamed

    def _has_home(self, module_name):
        """Whether the model class has somewhere to put this tensor."""
        known = self._homes.get(module_name)
        if known is not None:
            return known
        parent, _, leaf = module_name.rpartition(".")
        try:
            owner = self.model.get_submodule(parent) if parent else self.model
            found = hasattr(owner, leaf)
        except AttributeError:
            found = False
        self._homes[module_name] = found
        return found

    def _drop_homeless(self, state_dict):
        """Leave out checkpoint tensors this model class has nowhere to put.

        A checkpoint outlives the library that wrote it. transformers 5 stopped keeping a rotary
        embedding on each attention module, so every Llama checkpoint written before that carries
        ``...self_attn.rotary_emb.inv_freq`` for a module the model no longer builds -- and placing
        it raised an AttributeError from inside accelerate naming an attribute, which says nothing
        about which checkpoint tensor caused it or that the checkpoint is simply older than the
        library.

        Dropping it is right: inv_freq is derived from the config, not learned, and a model class
        that does not declare it computes it for itself. Dropping something the model DOES need is
        not silent either -- that parameter stays on meta and the forward fails on it by name.
        """
        keep = OrderedDict()
        homeless = []
        for name, value in state_dict.items():
            if self._has_home(name):
                keep[name] = value
            else:
                homeless.append(name)
        if homeless:
            caps.announce_once(
                "checkpoint-homeless-tensors",
                f"this checkpoint stores tensors the model class does not build, so they are not "
                f"placed: {', '.join(sorted(homeless)[:3])}"
                f"{' and others' if len(homeless) > 3 else ''}. This is what an older checkpoint "
                f"looks like to a newer transformers; anything the model actually needs would fail "
                f"by name at the first forward rather than be skipped here.", logging.INFO)
        return keep

    def submodule(self, checkpoint_name):
        """The module the checkpoint stores under this name."""
        model_attr = self.model
        for attr_name in self.module_name(checkpoint_name).split("."):
            model_attr = getattr(model_attr, attr_name)
        return model_attr

    def has_submodule(self, checkpoint_name):
        try:
            self.submodule(checkpoint_name)
        except AttributeError:
            return False
        return True

    def set_layers_from_layer_names(self):
        self.layers = [self.submodule(self.layer_names_dict["embed"])]
        self.layers.extend(list(self.submodule(self.layer_names_dict["layer_prefix"])))
        self.layers.append(self.submodule(self.layer_names_dict["norm"]))
        self.layers.append(self.submodule(self.layer_names_dict["lm_head"]))

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

        The one choke point where a shard's tensor names are turned into the model's parameter
        names. Everything downstream of here -- the quant plan, the coalesced binds, the list of
        names sent back to meta afterwards -- is in the model's namespace and needs no further
        translation.
        """
        state_dict = self._prepare_layer(self._drop_homeless(
            self._to_module_namespace(state_dict)))

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
            # The dtype is stated rather than left to default, and that is not belt-and-braces.
            # Given no dtype, accelerate casts the value to whatever the existing parameter is --
            # which is the meta placeholder built from the config. That silently reverses the
            # decision the registry already made in _plan_transfer, AND allocates a copy, which is
            # exactly what coalescing into one buffer exists to avoid. Passing the group's own
            # target keeps the bind a view.
            set_module_tensor_to_device(self.model, param_name, self.running_device, value=view,
                                        dtype=target_dtype)
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

        A multimodal checkpoint carries a vision tower and a projector; some architectures add extra
        top-level norms. None of them ever gets a streaming hook, so without this they stay on the
        meta device and fail the moment they run.

        Resident rather than streamed, and that is a decision rather than an oversight. A vision
        tower runs once per REQUEST, not once per token, so streaming it would add a full pass of
        its bytes to every prompt to buy back device memory the decoder is not contending for in
        between. What it does cost is measured and reported, and the weight cache is sized after
        this runs -- the budget it reads is what is left once these are placed, so the two cannot
        double-count.
        """
        self.resident_bytes = 0
        placed = []
        for name in self.layer_names_dict.get('resident', []):
            # Asked BEFORE the read, not after. These groups are discovered from the checkpoint
            # rather than declared, so a checkpoint routinely carries one this model class does not
            # build -- a multi-token-prediction head is the common case, and Qwen3.5 ships an 849MB
            # one. Reading it first meant paying for the whole shard, in time and in host memory,
            # to then throw it away.
            if not self.has_submodule(name):
                caps.announce_once(
                    f"resident-absent-{name}",
                    f"the checkpoint stores {name}, which this model class does not build, so "
                    f"those tensors are not read at all. Nothing runs them; this is not an error.",
                    logging.INFO)
                continue
            try:
                state_dict = self.load_layer_to_cpu(name)
            except FileNotFoundError:
                # Not every checkpoint of a given architecture ships every optional module.
                continue
            self.move_layer_to_device(state_dict)
            self.resident_bytes += self._shard_bytes(name)
            placed.append(name)
        if placed:
            print(f"resident modules: {', '.join(placed)} -- "
                  f"{self.resident_bytes / 1024 ** 2:.1f}MB placed once and kept outside the "
                  f"streaming cache.")

    def _shard_bytes(self, name):
        """Packed bytes of one module's shard, from its header rather than its data."""
        try:
            return self.loader.plan(name).total_bytes
        except Exception:  # noqa: BLE001 - a byte count must not be able to fail a load
            return 0

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
        #: Where a forward pass begins, so the device budget is sampled once per token.
        self._first_streamed = self._streamed_indices[0] if self._streamed_indices else None

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
        #: Cache key -> the checkpoint tensors that unit is made of, and what they cost packed. A
        #: unit here is anything the mixture streams on its own: one routed expert, or one always-on
        #: module beside them. Both are ordinary cache entries; only their policy differs.
        self._unit_tensor_keys = {}
        self._unit_byte_counts = {}
        self.experts = ExpertResidency(
            aging_interval=self._knob("expert_aging_interval", 4096),
            replan_interval=self._knob("expert_replan_interval", 512),
            enabled=self.expert_residency == "auto")

        # Both layouts read individual tensors -- or individual rows of one -- out of a shard, which
        # is a safetensors capability. Other persisters store something this cannot seek into.
        if type(ModelPersister.get_model_persister()).__name__ != 'SafetensorModelPersister':
            return

        layer_prefix = self.layer_names_dict['layer_prefix']
        hooked_experts = 0
        fused_containers = 0
        shared_modules = 0
        skipped = []

        for idx in self._streamed_indices:
            layer_name = self.layer_names[idx]
            if not layer_name.startswith(layer_prefix + '.'):
                continue
            try:
                shapes, packed_bytes = self._layer_tensor_specs(layer_name)
            except Exception:  # noqa: BLE001 - a layer we cannot inspect simply streams whole
                continue

            layout = detect_expert_layout(self.layers[idx], shapes, layer_name, self.config,
                                          conversion=self.conversion)
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
                        shared_modules += self._hook_shared_modules(idx, container, packed_bytes)
                    continue
                count = self._hook_expert_modules(idx, container, packed_bytes)
                if count:
                    accepted.append(container)
                    hooked_experts += count
                    self._hook_router(idx, layer_name, container)
                    shared_modules += self._hook_shared_modules(idx, container, packed_bytes)

            if not accepted:
                continue
            # Recomputed rather than taken from the layout, because a container detection was happy
            # with may still have been rejected just above; its tensors have to go back to the
            # layer's own stream or nothing would ever load them.
            owned = {key for container in accepted for key in container.owned_keys}
            self._non_expert_keys[idx] = [key for key in shapes if key not in owned]
            self._expert_layouts[idx] = ExpertLayout(containers=tuple(accepted),
                                                     other_keys=tuple(self._non_expert_keys[idx]))

        self._expert_streaming = bool(hooked_experts or fused_containers)
        self._report_expert_streaming(hooked_experts, fused_containers, shared_modules, skipped)

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
            size = self._register_unit(key, keys, packed_bytes, module)
            self.experts.track_expert(key, idx, expert_idx, size)
            hooked += 1
        return hooked

    def _register_unit(self, key, keys, packed_bytes, module):
        """Make one module a cache entry of its own. Returns what it costs packed."""
        size = sum(packed_bytes.get(name, 0) for name in keys)
        self._unit_tensor_keys[key] = keys
        self._unit_byte_counts[key] = size
        module._rocketllm_unit = key
        module.register_forward_pre_hook(self._unit_pre_hook)
        module.register_forward_hook(self._unit_post_hook)
        return size

    def _hook_shared_modules(self, idx, container, packed_bytes):
        """Give each always-on module beside the experts its own entry. Returns how many.

        These are the shared experts. Streaming them with the layer would be correct but wasteful in
        the other direction: they are read on every token, so they are the first thing that should
        be resident, and bundling them into the layer's entry means the pin planner cannot keep the
        small always-on parts without also keeping a large shared feed-forward it may have no room
        for. Split out, they rank in their own class -- above every routed expert, below attention.
        """
        hooked = 0
        for path, keys in container.shared_keys.items():
            try:
                module = self.layers[idx].get_submodule(path)
            except AttributeError:
                continue
            key = (idx, shared_kind(path))
            self.experts.track_shared(key, self._register_unit(key, keys, packed_bytes, module))
            hooked += 1
        return hooked

    def _hook_router(self, idx, layer_name, container):
        """Watch a layer's router, so its top-k is known the moment it is chosen.

        This is the only lookahead a mixture allows. Layer L's router runs inside layer L, so the
        instant it fires is the earliest anything can know which experts the layer needs -- and,
        equally, the latest, because layer L+1's routing does not exist yet and never will until L
        has finished. Returns the selection slot, or None when the router cannot be read, in which
        case the layer simply loads its experts as they run.
        """
        if not container.router_path or container.top_k is None:
            return None
        try:
            router = self.layers[idx].get_submodule(container.router_path)
        except AttributeError:
            return None
        selection = RouterSelection(container.num_experts, container.top_k,
                                    path=f"{layer_name}.{container.path}")
        router._rocketllm_router = (idx, selection)
        router.register_forward_hook(self._router_hook)
        return selection

    def _hook_fused_container(self, idx, layer_name, container):
        """Watch the router, then stream the rows it chose into the fused tensor."""
        try:
            experts = self.layers[idx].get_submodule(container.path)
        except AttributeError:
            return False

        selection = self._hook_router(idx, layer_name, container)
        if selection is None:
            # This layout cannot stream without the selection; detection only offers a fused
            # container when the router was identified, so reaching here means the module moved.
            return False

        experts._rocketllm_fused = (idx, container, selection)
        experts.register_forward_pre_hook(self._fused_pre_hook)
        experts.register_forward_hook(self._fused_post_hook)
        return True

    def _report_expert_streaming(self, hooked_experts, fused_containers, shared_modules, skipped):
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
        if shared_modules:
            shared_bytes = self.experts.report()["shared_bytes"]
            print(f"shared experts: {shared_modules} always-on modules "
                  f"({shared_bytes / 1024 ** 2:.1f}MB) are pinned ahead of any routed expert, "
                  f"because every token reads them.")
        if self._expert_streaming:
            print(f"expert residency: popularity counted from the router's own selection, halved "
                  f"every {self.experts.stats.aging_interval} firings, pins re-ranked every "
                  f"{self.experts.replan_interval}.")
        for reason in dict.fromkeys(reason for _, reason in skipped):
            print(f"expert container streamed with its layer instead: {reason}")

    def expert_report(self):
        """What the mixture did, for the benchmark. ``None`` when the model has no experts."""
        residency = getattr(self, "experts", None)
        if residency is None:
            return None
        report = residency.report()
        return report if report.get("experts") else None

    def _unit_pre_hook(self, module, args):
        """Hand one expert, or one always-on module, to the cache as a dense module's hook does.

        Going through the cache rather than reading the shard here is what stops a mixture paying
        for the same expert on every token. One still resident from the last token costs nothing;
        one that is not is read, and then competes for residency on its measured popularity -- the
        LFU half of the cache's policy, which exists for precisely this.

        By the time this runs the read has usually already been started: the router fired earlier in
        this same layer and the whole top-k went to the prefetch workers together, so what happens
        here is collecting a read in flight rather than beginning one.
        """
        self.cache.acquire(module._rocketllm_unit)

    def _unit_post_hook(self, module, args, output):
        self.cache.release(module._rocketllm_unit)
        return output

    def _router_hook(self, module, args, output):
        """A layer just routed. Start those reads and count the choice."""
        idx, selection = module._rocketllm_router
        selected = selection.observe(output)
        if selected:
            self.experts.on_router(idx, selected)
        return output

    def _fused_pre_hook(self, module, args):
        idx, container, selection = module._rocketllm_fused
        layer_name = self.layer_names[idx]
        rows = selection.take()
        if rows is None:
            # The router's choice could not be read. Every row is always correct, and costs what a
            # whole layer has always cost -- which is the behaviour this replaces, not a regression.
            if container.is_merged:
                # There is no whole-layer path here: the per-expert tensors are not parameters of
                # anything, so they cannot simply be placed. Every row is assembled instead, which
                # is the same work in the same shape, for all of the experts rather than a few.
                rows = tuple(range(container.num_experts))
            else:
                state_dict = load_layer_subset(self.checkpoint_path, layer_name, container.keys)
                module._rocketllm_moved = self.move_layer_to_device(state_dict)
                return
        if container.is_merged:
            compact = self._merge_expert_rows(container, layer_name, rows)
        else:
            compact = load_layer_rows(self.checkpoint_path, layer_name,
                                      {key: rows for key in container.keys})
        module._rocketllm_moved = self._place_expert_rows(
            container, rows, compact, container.target_shapes(layer_name))

    def _merge_expert_rows(self, container, layer_name, rows):
        """Build one compacted tensor per fused parameter out of the routed experts' own tensors.

        This is the transformers-5 shape, where the module is batched and the checkpoint is not. It
        reads the same bytes the fused layout's row slicing reads -- a per-expert checkpoint already
        stores each expert separately, so asking for the routed few asks for their bytes and no
        others -- and then performs the join the model declares, one row at a time.

        The result is keyed and shaped exactly as a slice out of a natively fused checkpoint would
        be, so everything downstream is the path that was already there.
        """
        wanted = []
        for target in container.merged_targets.values():
            for expert in rows:
                wanted.extend(target.sources[expert])
        # dict.fromkeys rather than a set: a tensor asked for twice must be read once, and the read
        # order should still follow the file rather than a hash.
        shard = load_layer_subset(self.checkpoint_path, layer_name, dict.fromkeys(wanted))

        compact = {}
        for target in container.merged_targets.values():
            built = []
            for expert in rows:
                parts = [shard[key] for key in target.sources[expert]]
                row = parts[0] if len(parts) == 1 else torch.cat(parts, dim=target.concat_dim)
                built.append(row.unsqueeze(0))
            compact[f"{layer_name}.{target.path}"] = (built[0] if len(built) == 1
                                                      else torch.cat(built, dim=0))
        return compact

    def _fused_post_hook(self, module, args, output):
        for param_name in getattr(module, '_rocketllm_moved', []):
            set_module_tensor_to_device(self.model, param_name, 'meta')
        module._rocketllm_moved = []
        return output

    def _place_expert_rows(self, container, rows, compact, shapes):
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
            full = torch.zeros(shapes[key], dtype=target, device=self.running_device)
            # `key` is the checkpoint's name, which is what the shard was read by and what
            # fused_shapes is keyed on; the model may keep it somewhere else. This path does not go
            # through move_layer_to_device, so it does its own translation.
            param_name = self.module_name(key)
            self._adopt_checkpoint_shape(param_name, full)
            full.index_copy_(0, index, value.to(device=self.running_device, dtype=target))
            set_module_tensor_to_device(self.model, param_name, self.running_device, value=full)
            moved.append(param_name)
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
            overridden = max(0, total - int(self._overrides["reserve_bytes"]))
            # The window budget is a *share* of usable memory, so it has to move with it. Left at
            # the value derived for the whole card it can exceed the overridden device budget
            # outright, and since pinning gets what the window does not, the pin budget then
            # collapses to zero and nothing is ever kept -- an override for a smaller card would
            # silently disable residency rather than emulating one.
            if device_bytes > 0:
                window_budget = int(window_budget * overridden / device_bytes)
            device_bytes = overridden

        # The profile's figure was measured on an idle card before this model existed. What is
        # actually free now, with the weights already placed, is what the budget is reading, so
        # prefer it -- on any backend that can report it honestly. Sizing the cache against the
        # older, larger number is how a run ends up pinning more than the card has left.
        live = self.budget.measure()
        if not live.estimated:
            window_share = (window_budget / device_bytes) if device_bytes else 0.0
            device_bytes = live.usable
            window_budget = int(device_bytes * window_share)
        self._report_resident_share(device_bytes)
        pinned = self._plan_pins(device_bytes, window_budget, largest)

        self.cache = TieredWeightCache(
            fetch=self._cache_fetch, sizer=self._packed_bytes,
            to_device=self._cache_to_device, to_host=self._cache_to_host,
            discard=self._cache_discard,
            sequence=self._cache_sequence if self.prefetching else None,
            device_bytes=device_bytes, host_bytes=host_bytes, window=window, pinned=pinned,
            profile=self.profile, prefetch_workers=self._knob("io_workers", 1))

        # The mixture needs the cache to prefetch into and to hand new pin plans to, and the cache
        # needs the pin plan the mixture cannot produce until something has routed. They are wired
        # to each other here, once both exist.
        self.experts.cache = self.cache
        self.experts.on_replan(self._replan_expert_pins)

        # The window is a share of the budget, so when the budget moves the window has to move with
        # it. Kept as the ratio rather than the bytes for exactly that reason.
        self._window_share = (window_budget / device_bytes) if device_bytes else 0.0
        # What the budget publishes is what the CACHE may hold, so it has to count what the cache
        # is already holding as its own. Wired here rather than at construction because the budget
        # has to exist first: it is what sized the cache.
        self.budget.reclaimable = self._cache_holdings
        self.budget.on_change = self._budget_changed

        self._resolve_kv_cache(device_bytes)

        print(f"cache: window {window} layers, device budget "
              f"{device_bytes / 1024 ** 3:.1f}GB, host tier {host_bytes / 1024 ** 3:.1f}GB, "
              f"{len(pinned)} pinned, pin_policy={self.pin_policy}")

    def _report_resident_share(self, device_bytes):
        """Say so when the modules held outside the cache took a real bite out of the device.

        A vision tower is placed before the cache is sized, so by the time the budget is read those
        bytes are simply gone and nothing downstream would ever mention them. On a card with room
        that is invisible and fine; on one without, it is the whole explanation for why the decoder
        can keep so few layers, and it has to be said once rather than left to be inferred from a
        low hit rate. The threshold below changes no behaviour -- it decides only whether the line
        is worth printing -- so it is a reporting choice rather than a tuning constant.
        """
        resident = getattr(self, "resident_bytes", 0)
        # A backend with no device pool to speak of -- CPU, or one whose driver reports nothing --
        # has no share to report, and dividing by what it did report would say "100%" about a
        # number that means nothing. The cache prints the budget it actually got either way.
        if not resident or device_bytes <= 0:
            return
        share = resident / (resident + device_bytes)
        if share <= 0.25:
            return
        caps.announce_once(
            "resident-share",
            f"modules held resident outside the weight cache take "
            f"{resident / 1024 ** 3:.2f}GB, {share * 100:.0f}% of this device's usable memory, "
            f"leaving {device_bytes / 1024 ** 3:.2f}GB for streamed weights and the KV cache. On a "
            f"vision-language checkpoint that is the vision tower; it runs once per request rather "
            f"than once per token, so it is kept rather than streamed. If the decoder ends up "
            f"holding too few layers to be worth it, a text-only checkpoint of the same family "
            f"will run faster on this device.", logging.WARNING)

    def _resolve_kv_cache(self, device_bytes):
        """Settle the KV cache now that both the weights and the budget have been measured."""
        weight_bytes = sum(self._layer_packed_bytes(idx) for idx in self._streamed_indices)
        weight_bytes += sum(self._unit_byte_counts.values())
        # Kept because speculation needs the same pair of numbers: whether the weights are resident
        # decides both what the context should cost and whether amortizing a pass buys anything.
        self._weight_bytes = weight_bytes
        self._device_budget = device_bytes
        headroom = self._knob("kv_fit_headroom_percent", 15) / 100.0
        self.kv_cache_choice, reason = resolve_kv_cache(
            self.kv_cache_setting, weight_bytes=weight_bytes, device_bytes=device_bytes,
            headroom=headroom)
        self.kv_cache_config = KVCacheConfig(
            group_size=self._knob("kv_group_size", 64),
            residual_length=self._knob("kv_residual_tokens", 128),
            compute_dtype=self.running_dtype)
        print(f"kv cache: {self.kv_cache_choice} -- {reason}")

    def _new_kv_cache(self):
        """A fresh cache for one generation, or None to let transformers use its own."""
        return build_kv_cache(self.kv_cache_choice, self.kv_cache_config,
                              device=self.running_device, compute_dtype=self.running_dtype)

    # ---- speculative decoding ----------------------------------------------------------------

    def _setup_speculation(self):
        """Load the draft, tell the budget it took memory, and build the decoder.

        Ordered that way on purpose. The draft is resident by definition, so its bytes come out of
        the same pool the weight cache was just sized against; registering them republishes the
        budget at once and the cache shrinks to fit alongside it. Discovering the shortfall later,
        through an ordinary sample, would mean a generation starting with the card over-committed.
        """
        enabled, reason = resolve_speculation(
            self.speculative_setting, self.draft_path, self.profile,
            weight_bytes=getattr(self, "_weight_bytes", None),
            device_bytes=getattr(self, "_device_budget", None))
        if not enabled:
            announce_speculation(False, reason)
            if self.draft_path:
                print(f"speculation: off -- {reason}")
            return

        self.draft = DraftModel.load(self.draft_path, target_config=self.config,
                                     target_tokenizer=self.tokenizer, device=self.running_device,
                                     dtype=self.running_dtype, hf_token=self.hf_token,
                                     trust_remote_code=self.trust_remote_code)
        draft_bytes = self.draft.device_bytes()
        self.budget.register_external("draft_model", draft_bytes)

        ceiling = lookahead_ceiling(self.profile)
        self.spec = SpeculativeDecoder(
            self.model, self.draft, lookahead=ceiling, max_lookahead=ceiling,
            new_cache=self._new_spec_cache,
            eos_token_id=getattr(self.generation_config, "eos_token_id", None))
        announce_speculation(True, reason, self.draft.name, ceiling)
        print(f"speculation: on, draft {self.draft.name} resident "
              f"({draft_bytes / 1024 ** 3:.2f}GB), up to {ceiling} tokens per pass -- {reason}")

    def _new_spec_cache(self):
        """A verification cache. Never None: the decoder has to be able to crop it."""
        from transformers.cache_utils import DynamicCache

        return self._new_kv_cache() or DynamicCache()

    def _speculative_call(self, args, kwargs):
        """The arguments for a speculative run, or None when this call is not one it can serve.

        Speculation replaces transformers' generation loop, so anything that loop does which this
        one does not has to fall back rather than be silently ignored. Returning None is that
        fallback, and the list below is deliberately conservative: an unsupported option produces
        the stock path and the right answer, never a quietly different generation.
        """
        if self.spec is None:
            return None
        non_text = sorted(name for name in kwargs
                          if any(marker in name for marker in _NON_TEXT_INPUT_MARKERS))
        if non_text:
            # The draft proposes from token ids alone. Handed a prompt whose images live in a
            # separate tensor it never sees, it would propose fluently about the wrong picture and
            # the verifier -- which does see them -- would reject nearly everything, so this is a
            # correctness stop as much as a throughput one.
            caps.announce_once(
                "spec-multimodal",
                f"this request carries non-text model inputs ({', '.join(non_text)}), which the "
                f"speculative loop does not propose over, so it runs on the ordinary generation "
                f"path. The output is the same as it would be without a draft model.",
                logging.INFO)
            return None
        input_ids = kwargs.get("input_ids", kwargs.get("inputs"))
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2:
            return None
        if input_ids.shape[0] != 1:
            return None

        config = self.model.generation_config
        unsupported = {
            "num_beams": (kwargs.get("num_beams") or getattr(config, "num_beams", 1) or 1) > 1,
            "num_return_sequences": (kwargs.get("num_return_sequences")
                                     or getattr(config, "num_return_sequences", 1) or 1) > 1,
            "penalties": any(kwargs.get(name) is not None for name in
                             ("repetition_penalty", "no_repeat_ngram_size", "bad_words_ids",
                              "logits_processor", "stopping_criteria", "assistant_model",
                              "prefix_allowed_tokens_fn", "constraints", "force_words_ids")),
        }
        blocked = [name for name, hit in unsupported.items() if hit]
        if blocked:
            caps.announce_once(
                "spec-fallback",
                f"this generate() call uses {', '.join(blocked)}, which the speculative loop does "
                f"not implement, so it runs on the ordinary generation path. The output is the "
                f"same as it would be without a draft model.", logging.INFO)
            return None

        max_new = kwargs.get("max_new_tokens")
        if max_new is None:
            max_length = kwargs.get("max_length") or getattr(config, "max_length", None)
            max_new = (max_length - input_ids.shape[1]) if max_length else None
        if max_new is None:
            max_new = getattr(config, "max_new_tokens", None)
        if not max_new or int(max_new) <= 0:
            return None

        sampling = SamplingParams.from_generation(
            config, do_sample=kwargs.get("do_sample"), temperature=kwargs.get("temperature"),
            top_k=kwargs.get("top_k"), top_p=kwargs.get("top_p"))
        # The streamer and an incoming cache both travel with the call rather than being listed as
        # unsupported above. They have to: a streamed request against a reused prefix is exactly
        # what an agentic conversation is, and refusing either would mean speculation never runs for
        # the server at all -- silently, since the fallback produces the same tokens.
        cache = kwargs.get("past_key_values")
        if cache is not None and not hasattr(cache, "crop"):
            # A rejected proposal has to come back out of the cache; without crop it cannot.
            return None
        return input_ids, int(max_new), sampling, kwargs.get("streamer"), cache

    def speculation_report(self):
        return self.spec.stats.to_dict() if self.spec is not None else None

    def _budget_changed(self, old_bytes, new_bytes, sample):
        """The published budget moved, so re-size the cache and re-rank what it keeps.

        This is the path a dropped KV cache travels down. At the end of a generation the context is
        released, the budget jumps by its whole size, and without this nothing would notice: the
        cache would go on sizing itself against whatever was free while the longest context of the
        run was still resident. The window keeps its share and the rest goes to pinning, so the
        memory a shorter context frees turns directly into resident weights.

        The new plan is adopted BEFORE the cache is resized, and the order is load-bearing when the
        budget has shrunk. Pinned entries are never evicted, so resizing first would ask the cache
        to fit inside a smaller capacity while still holding every pin the larger one paid for, and
        it would quietly fail to get there. Unpinning first is what lets the eviction actually free
        the memory the new budget says is no longer ours.
        """
        cache = getattr(self, "cache", None)
        if cache is None:
            return
        if sample is not None and sample.estimated:
            # The backend could not report the allocator's own totals, so this reading is free
            # memory and nothing else -- on the CPU backend that is host RAM, which the profile
            # deliberately does not treat as a device pool. Following it would silently replace a
            # derived budget with an unrelated number. Track, do not act.
            return
        capacity = max(0, int(new_bytes))
        window_budget = int(capacity * self._window_share)
        self._pin_budget = pin_budget_from(capacity, window_budget)
        if self.pin_policy != "off":
            cache.apply_plan(plan_pins(self._pin_candidates(), self._pin_budget))
        cache.resize_device(capacity)

    def _cache_holdings(self):
        """Device bytes the weight cache is already holding, for the budget to count as its own.

        Not free memory, and not something the budget could measure for itself: the driver reports
        these bytes as in use, which they are, but they are in use by the very thing being sized and
        it can hand them back. Sizing the cache to free memory alone would tell a cache holding a
        gigabyte of pinned weights that its budget is the few hundred megabytes left over, and it
        would evict most of itself to get there, on every sample, for nothing.
        """
        cache = getattr(self, "cache", None)
        return cache.device.bytes if cache is not None else 0

    def _plan_pins(self, device_bytes, window_budget, largest):
        """Which modules to keep resident for the whole run.

        Dense modules are touched once per token, so they all sit in the same priority class and the
        ranking reduces to size: pinning the smallest first keeps the most of them, which is what
        buys the hit rate. A mixture adds two more classes below that -- its always-on modules, then
        its routed experts ranked by the popularity actually observed -- and the class boundaries
        are absolute, so no expert displaces the attention block every token reads.

        At load nothing has routed yet, so the expert half of that list is empty and the plan is
        dense-only. It is rebuilt from real counts once the mixture has been running; see
        :meth:`_replan_expert_pins`.
        """
        self._pin_budget = pin_budget_from(device_bytes, window_budget)
        if self.pin_policy == "off":
            return ()
        return plan_pins(self._pin_candidates(), self._pin_budget).pinned

    def _pin_candidates(self):
        candidates = [PinCandidate(key=(idx, KIND_DENSE),
                                   packed_bytes=self._layer_packed_bytes(idx),
                                   priority=CLASS_ALWAYS, accesses_per_token=1.0)
                      for idx in self._streamed_indices]
        candidates.extend(self.experts.pin_candidates())
        return candidates

    def _replan_expert_pins(self):
        """Re-rank residency now that routing has been watched for a while.

        Called on a cadence the profile derives rather than on every firing: acting on a ranking
        that has barely moved costs an eviction and a re-read, which is more than the residency it
        buys back. The budget itself does not change here -- only who deserves it.
        """
        if self.pin_policy == "off" or getattr(self, "cache", None) is None:
            return None
        plan = plan_pins(self._pin_candidates(), self._pin_budget)
        self.cache.apply_plan(plan)
        return plan

    def _load_streamed_layer(self, idx):
        """Load one streamed module's weights. Experts are excluded when they stream themselves."""
        keys = self._non_expert_keys.get(idx) if getattr(self, '_expert_streaming', False) else None
        if keys is None:
            return self.load_layer_to_cpu(self.layer_names[idx])
        return load_layer_subset(self.checkpoint_path, self.layer_names[idx], keys)

    # ---- the cache's view of a streamed module ----------------------------------------------

    def _cache_fetch(self, key):
        """Storage tier: read a unit's tensors. Runs on a prefetch worker, so it only reads."""
        keys = self._unit_tensor_keys.get(key)
        if keys is not None:
            return _ResidentModule(key, host=load_layer_subset(
                self.checkpoint_path, self.layer_names[key[0]], keys))
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
        """Host tier: unbind from the module but keep the CPU copy to serve the next hit.

        Returns None when there is no CPU copy to keep, which sends the entry to storage instead.
        That happens to an entry that was pinned when it was placed -- pinned entries never fall
        back, so holding their host copy is pure waste and it is dropped -- and which a later plan
        then unpinned. Nothing could unpin an entry until residency became dynamic, which is why
        this could not arise before and why it must be handled now.
        """
        self._unbind(resident)
        return resident if resident.host is not None else None

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
        """Packed bytes of one cached unit: a dense module, an expert, or an always-on module."""
        size = self._unit_byte_counts.get(key)
        if size is not None:
            return size
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
        # Once per forward, at the first streamed module, rather than at every layer boundary. The
        # driver query behind this is not free -- sampling per layer cost ~20% of decode throughput
        # on a small model, which is the whole saving a budget this precise could ever buy back --
        # and it measures nothing extra: what the reading tracks is the KV cache growing and that
        # happens per token, not per layer.
        if module._rocketllm_idx == self._first_streamed:
            self.budget.sample()
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
        # The context has just been dropped, so the budget is about to jump by its whole size.
        # Republish immediately rather than waiting for the hysteresis streak: this is the one
        # moment the change is known to be real, and making it wait would leave the weight cache
        # sized for a context that no longer exists.
        budget = getattr(self, "budget", None)
        if budget is not None:
            budget.reset()

    def close(self):
        """Shut the model down for good: stop the prefetch worker and release everything."""
        draft = getattr(self, "draft", None)
        if draft is not None:
            # Withdraw the claim before dropping the weights, so the budget's last reading is taken
            # with the memory genuinely back rather than still attributed to something gone.
            self.budget.register_external("draft_model", 0)
            draft.reset()
            self.draft = None
            self.spec = None
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
        # Speculation replaces the generation loop rather than sitting inside it, so it is decided
        # before anything else is set up. A call it cannot serve exactly falls through to the stock
        # path; see _speculative_call for what that means and why the list is conservative.
        speculative = self._speculative_call(args, kwargs)
        if speculative is not None:
            input_ids, max_new, sampling, streamer, cache = speculative
            try:
                return self.spec.generate(input_ids, max_new, sampling, streamer=streamer,
                                          past_key_values=cache)
            finally:
                self._end_generation()

        # A quantized context has to be handed in, because transformers builds its own DynamicCache
        # otherwise and nothing here would ever see it. A caller who passed one keeps it: an
        # explicit cache is an instruction, not a suggestion. At fp16 this adds nothing at all --
        # build_kv_cache returns None and the stock path runs untouched.
        if kwargs.get("past_key_values") is None and self.kv_cache_choice != KV_FP16:
            cache = self._new_kv_cache()
            if cache is not None:
                kwargs["past_key_values"] = cache
                kwargs.setdefault("use_cache", True)
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
