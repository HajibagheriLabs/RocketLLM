"""Streaming benchmark harness.

Every optimization in this project is judged here. The headline number is not tokens per second --
that says more about the machine than about the engine -- it is BYTES MOVED PER TOKEN, BROKEN DOWN
BY MEMORY TIER:

    device    bytes served from weights already resident on the device (no read, no transfer)
    host      bytes pushed across the host->device link
    storage   bytes read out of the checkpoint on disk

Those three tell you two things at once: whether a change actually moved less data, and which regime
this particular machine is in. A model that fits in VRAM is a completely different problem from one
that is ten times the size of VRAM, and the same patch can help one and do nothing for the other.

Everything here is measured, not modelled. Byte counts come from wrapping the real loader and the
real device-placement call; timings come from wrapping the same paths. Nothing is inferred from a
tensor's declared size when the actual read can be observed instead.

Usage
-----
# Baseline a small model and write a result record:
python tests/bench_streaming.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --json

# Emulate a 4GB card:
python tests/bench_streaming.py --model Qwen/Qwen2.5-7B-Instruct --max-vram-gb 4 --json

# Diff against a previous record:
python tests/bench_streaming.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --compare-to bench_results/<previous>.json
"""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# Run from a plain checkout, with or without an editable install: the package sits at the repo
# root and cap_vram lives next door with the correctness harness, so put both within reach.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_streaming_gpu import cap_vram  # noqa: E402
from rocketllm.hw import HardwareProfile  # noqa: E402
from rocketllm.hw import caps  # noqa: E402
from rocketllm.quant import registry as quant_registry  # noqa: E402

RESULTS_DIR = REPO_ROOT / "bench_results"
SCHEMA_VERSION = 1

UNAVAILABLE = None  # a metric this backend cannot report; never silently reported as zero


# ---------------------------------------------------------------------------------------------
# capability queries
#
# These now come from rocketllm.hw.caps, which is the single place allowed to ask the machine
# anything. The bench keeps only the measurements that are scoped to one run -- peak memory and
# per-process read counters -- because those describe the run, not the hardware.
# ---------------------------------------------------------------------------------------------

pick_device = caps.resolve_device
device_sync = caps.synchronize
device_name = caps.device_name
supports_bf16 = caps.supports_bf16


def compute_capability(device):
    cc = caps.compute_capability(device)
    return f"{cc[0]}.{cc[1]}" if cc else UNAVAILABLE


def device_total_bytes(device):
    return caps.device_memory(device)[0]


def host_ram_total_bytes():
    return caps.host_memory()[0]


def peak_device_bytes(device):
    """Peak device allocation for this process, where the backend tracks one."""
    kind = device.type
    if kind == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated(device))
    if kind == "xpu" and hasattr(torch, "xpu"):
        try:
            return int(torch.xpu.max_memory_allocated(device))
        except Exception:
            return UNAVAILABLE
    # MPS exposes only a current figure, not a high-water mark, and CPU has no separate device
    # pool at all. Reporting either as 0 would read as "used no memory", which is a lie.
    return UNAVAILABLE


def reset_peak_device_stats(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        try:
            torch.xpu.reset_peak_memory_stats()
        except Exception:
            pass


def peak_host_rss_bytes():
    """Process peak resident set size, straight from the OS -- no sampling thread, no estimate."""
    if os.name == "nt":
        class _ProcMemCounters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = _ProcMemCounters()
        counters.cb = ctypes.sizeof(_ProcMemCounters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # A HANDLE is pointer-sized; without an explicit restype ctypes truncates it to int on
        # 64-bit and the call fails.
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        for library in ("psapi", "kernel32"):
            try:
                dll = ctypes.WinDLL(library, use_last_error=True)
                # Windows moved this into kernel32 as K32GetProcessMemoryInfo; try both names.
                func = getattr(dll, "GetProcessMemoryInfo", None) or \
                    getattr(dll, "K32GetProcessMemoryInfo", None)
                if func is None:
                    continue
                func.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcMemCounters), ctypes.c_ulong]
                func.restype = ctypes.c_int
                if func(kernel32.GetCurrentProcess(), ctypes.byref(counters),
                        ctypes.sizeof(counters)):
                    return int(counters.PeakWorkingSetSize)
            except (OSError, AttributeError):
                continue
        return UNAVAILABLE
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, the BSDs and macOS report bytes.
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, OSError):
        return UNAVAILABLE


def physical_read_bytes():
    """Bytes this process pulled through the block layer, where the OS exposes that.

    This is the one counter that separates a real disk read from a page-cache hit, which is the
    difference between the storage tier and the host tier. Linux exposes it; most other platforms
    do not, and there it is reported as unavailable rather than conflated with logical reads.
    """
    try:
        with open("/proc/self/io", "r") as handle:
            for line in handle:
                if line.startswith("read_bytes:"):
                    return int(line.split(":")[1].strip())
    except (OSError, ValueError):
        pass
    return UNAVAILABLE


def storage_backing(path):
    """Best-effort description of what the weights are sitting on."""
    try:
        if os.name == "nt":
            drive = os.path.splitdrive(os.path.abspath(str(path)))[0]
            return drive or UNAVAILABLE
        return f"dev:{os.stat(path).st_dev}"
    except OSError:
        return UNAVAILABLE


# ---------------------------------------------------------------------------------------------
# hardware profile
#
# The real thing now, from rocketllm.hw: measured bandwidths for every tier plus every derived
# tuning knob. A result without the machine it came from cannot be compared to anything, so the
# whole profile goes into the record -- including what the engine decided to do with it, since a
# future run that picks different knobs is not measuring the same configuration.
# ---------------------------------------------------------------------------------------------

def probe_hardware(device, weights_path=None, reprofile=False):
    profile = HardwareProfile.load_or_probe(weights_path=weights_path, device=str(device),
                                            reprofile=reprofile)
    as_dict = profile.to_dict()
    # `profile_key` is what the comparison guard and the result filename key off. Keeping the name
    # means records written before this module existed still line up.
    as_dict["profile_key"] = profile.fingerprint
    return as_dict


def hardware_identity(profile):
    """The fields that decide whether two runs happened on the same machine.

    Read from the profile rather than trusting its fingerprint: the fingerprint is a hash whose
    recipe can change between versions, and when it does, every stored record would suddenly look
    like a different machine. The underlying facts do not move like that.
    """
    def cc(value):
        if isinstance(value, (list, tuple)):
            return ".".join(str(part) for part in value)
        return str(value) if value is not None else None

    return {
        "backend": profile.get("backend"),
        # `device` is the older stub's spelling of the same field.
        "device": profile.get("device_name") or profile.get("device"),
        "device_total_bytes": profile.get("device_total_bytes"),
        "compute_capability": cc(profile.get("compute_capability")),
        "host_total_bytes": (profile.get("host_total_bytes")
                             or profile.get("host_ram_total_bytes")),
        "machine": (profile.get("versions") or {}).get("machine") or profile.get("machine"),
    }


def software_env():
    import transformers
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


# ---------------------------------------------------------------------------------------------
# meters
# ---------------------------------------------------------------------------------------------

class Meters:
    """Byte and time counters, written from the forward thread and the prefetch worker alike."""

    FIELDS = ("storage_bytes", "storage_seconds", "storage_reads",
              "transfer_bytes", "transfer_seconds", "transfers",
              "dequant_seconds", "compute_seconds", "storage_wait_seconds", "evict_seconds")

    def __init__(self):
        self._lock = threading.Lock()
        self._compute_started = {}
        self.storage_bytes_exact = True
        for field in self.FIELDS:
            setattr(self, field, 0 if field.endswith(("bytes", "reads", "transfers")) else 0.0)

    def add(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, getattr(self, key) + value)

    def snapshot(self):
        with self._lock:
            return {field: getattr(self, field) for field in self.FIELDS}

    def reset(self):
        with self._lock:
            for field in self.FIELDS:
                setattr(self, field, 0 if field.endswith(("bytes", "reads", "transfers")) else 0.0)


def diff_snapshots(later, earlier):
    return {key: later[key] - earlier[key] for key in later}


# ---------------------------------------------------------------------------------------------
# instrumentation
#
# Wrappers only. The streaming code is not restructured for the benchmark; every patch here calls
# straight through to the original and is removed again afterwards.
# ---------------------------------------------------------------------------------------------

class Instrumentation:
    def __init__(self, meters, device, sync_phases):
        self.meters = meters
        self.device = device
        self.sync_phases = sync_phases
        self._originals = []

    def _patch(self, obj, name, replacement):
        self._originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    def install(self):
        """Patch before the model is built.

        The streaming hooks are registered during construction and capture the bound method as it
        is *then*, so wrapping the hooks afterwards would have no effect on an existing model.
        """
        import rocketllm.base as base

        meters = self.meters
        sync = self._sync

        original_load_layer = base.load_layer
        original_load_subset = base.load_layer_subset
        original_place = base.set_module_tensor_to_device

        def load_layer(local_path, layer_name):
            start = time.perf_counter()
            state_dict = original_load_layer(local_path, layer_name)
            elapsed = time.perf_counter() - start

            shard = Path(local_path) / (layer_name + ".safetensors")
            try:
                # A whole-shard load reads the whole file, so the file itself is the exact number.
                read_bytes = os.path.getsize(shard)
            except OSError:
                # Another persister (or another on-disk layout) -- fall back to what came back and
                # flag the record, because that misses any container overhead.
                read_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
                meters.storage_bytes_exact = False
            meters.add(storage_bytes=read_bytes, storage_seconds=elapsed, storage_reads=1)
            return state_dict

        def load_layer_subset(local_path, layer_name, keys):
            start = time.perf_counter()
            result = original_load_subset(local_path, layer_name, keys)
            elapsed = time.perf_counter() - start
            # A subset read seeks to exactly these tensors, so their own size is what came off disk.
            read_bytes = sum(v.numel() * v.element_size() for v in result.values())
            meters.add(storage_bytes=read_bytes, storage_seconds=elapsed, storage_reads=1)
            return result

        def place(module, tensor_name, device, value=None, dtype=None, **kwargs):
            # device='meta' is an eviction: it frees, it does not transfer.
            evicting = str(device) == "meta"
            # A value already sitting on the target device is being *bound*, not moved -- that is
            # what the coalesced path does once its single transfer has landed. Counting its bytes
            # here would charge the link for data that never crossed it.
            binding = (not evicting and value is not None
                       and value.device.type == torch.device(str(device)).type)
            start = time.perf_counter()
            result = original_place(module, tensor_name, device, value=value, dtype=dtype, **kwargs)
            if evicting or binding:
                return result
            sync()
            elapsed = time.perf_counter() - start
            moved = 0
            if value is not None:
                itemsize = _itemsize(dtype) if dtype is not None else value.element_size()
                moved = value.numel() * itemsize
            meters.add(transfer_bytes=moved, transfer_seconds=elapsed, transfers=1)
            return result

        self._patch(base, "load_layer", load_layer)
        self._patch(base, "load_layer_subset", load_layer_subset)
        # Two modules place tensors: the base model binds the coalesced buffer's views, and the
        # quant registry places whatever could not be coalesced. Both hold their own reference to
        # accelerate's placer, so both have to be wrapped or the tier accounting quietly loses
        # every weight that took the individual path.
        self._patch(base, "set_module_tensor_to_device", place)
        self._patch(quant_registry, "set_module_tensor_to_device", place)

        model_cls = base.RocketModel
        original_prepare = model_cls._prepare_layer
        original_pre = model_cls._pre_hook
        original_post = model_cls._post_hook

        def prepare_layer(model_self, state_dict):
            # Whatever the checkpoint's format does to a shard before it can be placed. For an
            # unquantized model that is nothing; for a packed one it is the dequantization, which
            # is the only work worth calling a dequant phase.
            start = time.perf_counter()
            try:
                return original_prepare(model_self, state_dict)
            finally:
                meters.add(dequant_seconds=time.perf_counter() - start)

        def pre_hook(model_self, module, args):
            # Whatever the pre-hook spends beyond placing and dequantizing the weights is time
            # spent *obtaining* them: either blocking on the prefetch worker or reading inline.
            # That wait is the storage tier's real cost to the critical path, and with prefetching
            # on it is nowhere near the worker-thread read time -- so it has to be measured here
            # rather than inferred from what the loader reported.
            hook_started = time.perf_counter()
            before = meters.snapshot()
            result = original_pre(model_self, module, args)
            sync()
            hook_elapsed = time.perf_counter() - hook_started
            after = meters.snapshot()
            accounted = ((after["transfer_seconds"] - before["transfer_seconds"])
                         + (after["dequant_seconds"] - before["dequant_seconds"]))
            meters.add(storage_wait_seconds=max(0.0, hook_elapsed - accounted))

            # Everything from here until the post hook is the module's own forward.
            meters._compute_started[id(module)] = time.perf_counter()
            return result

        def post_hook(model_self, module, args, output):
            started = meters._compute_started.pop(id(module), None)
            if started is not None:
                sync()
                meters.add(compute_seconds=time.perf_counter() - started)
            # Releasing the layer is its own phase, and not a free one: the post hook sends the
            # weights back to meta and then collects. Folding it into "unattributed" would hide
            # the single largest cost in the current pipeline.
            evict_started = time.perf_counter()
            try:
                return original_post(model_self, module, args, output)
            finally:
                meters.add(evict_seconds=time.perf_counter() - evict_started)

        original_coalesced = getattr(model_cls, "_move_coalesced", None)

        def move_coalesced(model_self, entries, target_dtype):
            """The coalesced path stages a whole layer and sends it in one transfer.

            All of that -- packing the staging buffer, the buffer itself, the copy -- is the cost
            of getting the layer onto the device, so it is charged to the host->device phase. The
            binds that follow are free and are skipped by the placement wrapper above.
            """
            nbytes = sum(value.numel() for _, value in entries) * _itemsize(target_dtype)
            start = time.perf_counter()
            try:
                return original_coalesced(model_self, entries, target_dtype)
            finally:
                sync()
                meters.add(transfer_bytes=nbytes,
                           transfer_seconds=time.perf_counter() - start, transfers=1)

        self._patch(model_cls, "_prepare_layer", prepare_layer)
        if original_coalesced is not None:
            self._patch(model_cls, "_move_coalesced", move_coalesced)
        self._patch(model_cls, "_pre_hook", pre_hook)
        self._patch(model_cls, "_post_hook", post_hook)

    def _sync(self):
        if self.sync_phases:
            device_sync(self.device)

    def remove(self):
        for obj, name, original in reversed(self._originals):
            setattr(obj, name, original)
        self._originals = []


def _itemsize(dtype):
    try:
        return dtype.itemsize
    except AttributeError:  # torch < 2.1
        return torch.empty((), dtype=dtype).element_size()


def resident_bytes(model):
    """Bytes that live on the device between layers, and so are served without moving at all.

    Measured at steady state, before generation: whatever is resident then is re-read from device
    memory on every single token, which is exactly what the device tier means.
    """
    total = 0
    seen = set()
    for _, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor is None or tensor.device.type == "meta":
            continue
        if id(tensor) in seen:  # tied weights are one allocation, not two
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return total


# ---------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------

def make_phase_marker(meters):
    """Timestamps every generation step, separating prefill from decode.

    transformers calls the logits processors once per generated token, so the first call lands at
    the end of prefill and each later one at the end of a decode step. Riding on the callback beats
    running generation twice or reaching into the generation loop. Subclassing the real base class
    keeps it acceptable to generate()'s processor-list validation.

    Speculation does not go through that loop -- one verification pass emits several tokens and
    calls no logits processor at all -- so the mark also carries how many tokens it accounts for,
    and the decoder calls it directly. Without that count a speculative run would report passes per
    second under a heading that says tokens per second.
    """
    from transformers import LogitsProcessor

    class PhaseMarker(LogitsProcessor):
        def __init__(self):
            self.marks = []
            self.tokens = 0

        def mark(self, tokens=1):
            self.tokens += int(tokens)
            self.marks.append((time.perf_counter(), meters.snapshot(), self.tokens))

        def __call__(self, input_ids, scores):
            self.mark(1)
            return scores

    return PhaseMarker()


def load_model(args, device):
    """Build the streaming model, bypassing the platform override in AutoModel.

    AutoModel returns the MLX backend on macOS, which does not go through the streaming hooks this
    harness measures. Selecting the class directly keeps the bench pointed at the streaming engine
    on every platform; the MLX path needs its own harness.
    """
    import importlib
    from rocketllm.auto_model import AutoModel

    module_name, class_name = AutoModel.get_module_class(args.model)
    model_cls = getattr(importlib.import_module(module_name), class_name)
    print(f"streaming class: {class_name}")
    return model_cls(args.model, device=str(device), prefetching=not args.no_prefetch,
                     vram_reserve=args.vram_reserve, host_cache_gb=args.host_cache_gb,
                     expert_residency="off" if args.no_expert_residency else "auto",
                     draft_model=args.draft_model, speculative=args.speculative)


# ---- the conversation benchmark ------------------------------------------------------------------

#: Filler for the opening message, so the first prompt is long enough that prefill dominates the
#: turn. What it says does not matter; how many tokens it is does.
_FILLER = ("The engineer reviewed the streaming loader, the coalesced transfers, the tiered weight "
           "cache and the expert residency policy, then wrote down what each one measured. ")
_FOLLOW_UP = "Summarise the paragraph above in one sentence, then say what you would measure next."


def _conversation_prompt(model, turns_so_far, opening):
    """The prompt for the next turn, built the way a chat client builds one: everything again."""
    messages = [{"role": "user", "content": opening}]
    for reply in turns_so_far:
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": _FOLLOW_UP})
    tokenizer = model.tokenizer
    if getattr(tokenizer, "chat_template", None):
        try:
            encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                    tokenize=True, return_tensors="pt",
                                                    return_dict=True)
            return encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        except Exception:  # noqa: BLE001 - a template that cannot render this is not the point
            pass
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    return tokenizer([text], return_tensors="pt")["input_ids"]


def _one_turn(model, cache, input_ids, layers, device, new_tokens):
    """One turn, measuring what the prefill actually cost."""
    from rocketllm.server import prefix_cache as prefixes

    session = prefixes.PrefixSession(cache, input_ids[0].tolist(),
                                     config=getattr(model, "kv_cache_config", None),
                                     layers=layers, device=device)

    def new_cache():
        from transformers.cache_utils import DynamicCache

        built = model._new_kv_cache()
        return built if built is not None else DynamicCache()

    kv = session.begin(new_cache)
    timing = {}

    class Tap:
        """The streamer, reduced to the two things this needs: when the first token landed, and
        the ids, which the session keys its checkpoints by."""

        def __init__(self):
            self.prompt_seen = False

        def put(self, value):
            if not self.prompt_seen:
                self.prompt_seen = True
                return
            if "first_token" not in timing:
                timing["first_token"] = time.perf_counter()
            session.observe_tokens(value.reshape(-1).tolist())

        def end(self):
            pass

    device_sync(device)
    started = time.perf_counter()
    kwargs = dict(max_new_tokens=new_tokens, do_sample=False, use_cache=True, streamer=Tap(),
                  attention_mask=torch.ones_like(input_ids))
    if kv is not None:
        kwargs["past_key_values"] = kv
    sequences = model.generate(input_ids, **kwargs)
    device_sync(device)
    wall = time.perf_counter() - started

    prefill = timing.get("first_token", time.perf_counter()) - started
    session.finish(prefill_seconds=prefill)
    reply = model.tokenizer.decode(sequences[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return {
        "prompt_tokens": int(input_ids.shape[-1]),
        "tokens_reused": session.restored,
        "tokens_prefilled": session.prefilled,
        "prefill_seconds": prefill,
        "wall_seconds": wall,
        "hit": session.match is not None,
    }, reply


def run_conversation(args, device):
    """Replay a multi-turn conversation with the prefix cache on and off.

    The measurement that matters is the prefill of each turn, because that is the whole of what
    reuse can save: an agentic client resends everything it has, so on turn two onward the prefill
    is re-doing work the previous turn already paid a full streaming pass for.
    """
    from rocketllm.server import prefix_cache as prefixes

    model = load_model(args, device)
    layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    opening = _FILLER * max(1, args.conversation_filler)
    seed = prefixes.namespace_seed(args.model, getattr(model, "running_dtype", ""),
                                   getattr(model, "kv_cache_choice", ""), layers)

    warmup = _conversation_prompt(model, [], opening).to(device)
    print(f"opening prompt is {int(warmup.shape[-1])} tokens")

    runs = {}
    for label, enabled in (("prefix cache OFF", False), ("prefix cache ON", True)):
        # A throwaway generation before EACH mode, after the reset that precedes it. Without one,
        # the second mode's first turn also pays to read every weight back off storage, and that
        # one-off cost is an order of magnitude larger than the thing being compared -- measured at
        # 1.13s against a 0.10s warm prefill, which read as a tenfold regression that was not there.
        model.generate(warmup, max_new_tokens=2, do_sample=False, use_cache=True,
                       attention_mask=torch.ones_like(warmup))
        cache = prefixes.build(profile=getattr(model, "profile", None), seed=seed, enabled=enabled,
                               spill_dir=None)
        replies = []
        turns = []
        for _ in range(args.conversation):
            input_ids = _conversation_prompt(model, replies, opening).to(device)
            row, reply = _one_turn(model, cache, input_ids, layers, device,
                                   args.conversation_tokens)
            turns.append(row)
            replies.append(reply)
        runs[label] = {"turns": turns, "prefix_cache": cache.report()}
        model.reset()

    print()
    print("=" * 78)
    print(f"  {args.conversation}-TURN CONVERSATION  (prefill is what reuse can save)")
    print("=" * 78)
    for label, data in runs.items():
        print(f"\n  {label}")
        print(f"    {'turn':<6}{'prompt':>9}{'reused':>9}{'prefilled':>11}"
              f"{'prefill':>11}{'wall':>10}")
        for index, row in enumerate(data["turns"], start=1):
            print(f"    {index:<6}{row['prompt_tokens']:>9}{row['tokens_reused']:>9}"
                  f"{row['tokens_prefilled']:>11}{row['prefill_seconds']:>10.2f}s"
                  f"{row['wall_seconds']:>9.2f}s")
        report = data["prefix_cache"]
        print(f"    hits {report['hits']}/{report['lookups']}, "
              f"tokens skipped {report['tokens_skipped']}, "
              f"tokens prefilled {report['tokens_prefilled']}, "
              f"time saved vs a measured full prefill "
              f"{report['seconds_saved_vs_measured_baseline']:.2f}s")

    off = runs["prefix cache OFF"]["turns"]
    on = runs["prefix cache ON"]["turns"]
    print()
    print("  PREFILL PER TURN, measured")
    print(f"    {'turn':<6}{'off':>10}{'on':>10}{'change':>22}")
    for index, (a, b) in enumerate(zip(off, on), start=1):
        if a["prefill_seconds"] > 0:
            pct = (b["prefill_seconds"] - a["prefill_seconds"]) / a["prefill_seconds"] * 100.0
            change = f"{pct:+.1f}%  {'faster' if pct < -0.5 else 'slower' if pct > 0.5 else 'same'}"
        else:
            change = "n/a"
        print(f"    {index:<6}{a['prefill_seconds']:>9.2f}s{b['prefill_seconds']:>9.2f}s"
              f"{change:>22}")
    total_off = sum(row["prefill_seconds"] for row in off)
    total_on = sum(row["prefill_seconds"] for row in on)
    print(f"    {'total':<6}{total_off:>9.2f}s{total_on:>9.2f}s"
          f"{(total_on - total_off) / total_off * 100.0:>+21.1f}%" if total_off else "")
    print("=" * 78)
    return runs


def run(args, device):
    from transformers import LogitsProcessorList

    meters = Meters()
    instrumentation = Instrumentation(meters, device, sync_phases=not args.no_sync_phases)
    instrumentation.install()

    try:
        build_started = time.perf_counter()
        model = load_model(args, device)
        build_seconds = time.perf_counter() - build_started

        tokens = model.tokenizer([args.prompt], return_tensors="pt",
                                 return_attention_mask=False)["input_ids"].to(device)
        prompt_tokens = int(tokens.shape[-1])

        device_tier_bytes = resident_bytes(model.model)

        # Probe before the run, never after: the allocator probe resets peak memory stats, and the
        # storage sweep reads through the checkpoint. Both would corrupt the numbers if they ran
        # between generation and the point where those numbers are collected.
        hardware = probe_hardware(device, weights_path=getattr(model, "checkpoint_path", None),
                                  reprofile=args.reprofile)

        # Only the generation run is measured: the one-time checkpoint split and the resident
        # loads that precede it are setup, not per-token cost.
        meters.reset()
        reset_peak_device_stats(device)
        marker = make_phase_marker(meters)
        physical_before = physical_read_bytes()

        device_sync(device)
        started = time.perf_counter()
        if getattr(model, "spec", None) is not None:
            # The speculative loop is not transformers' loop, so it takes neither a logits
            # processor nor return_dict_in_generate. It marks its own passes instead, which is the
            # same signal from the only place that has it.
            model.spec.on_pass = marker.mark
            sequences = model.generate(tokens, max_new_tokens=args.max_new_tokens, do_sample=False,
                                       use_cache=True)
        else:
            sequences = model.generate(tokens,
                                       max_new_tokens=args.max_new_tokens,
                                       do_sample=False,
                                       use_cache=True,
                                       logits_processor=LogitsProcessorList([marker]),
                                       return_dict_in_generate=True).sequences
        device_sync(device)
        wall_seconds = time.perf_counter() - started

        physical_after = physical_read_bytes()
        totals = meters.snapshot()
        peak_device = peak_device_bytes(device)
        text = model.tokenizer.decode(sequences[0])
        new_tokens = int(sequences.shape[-1]) - prompt_tokens
    finally:
        instrumentation.remove()

    return build_record(args, device, meters, marker, totals, device_tier_bytes,
                        prompt_tokens, new_tokens, wall_seconds, build_seconds,
                        peak_device, physical_before, physical_after, text, model, hardware)


def cache_hit_rate(model):
    """Hit rate by tier, straight from the cache the run actually used.

    Reported per acquire rather than per byte, which is the number the replacement policy is
    steering: a hit is a layer that did not have to be read again, whatever it weighed.
    """
    cache = getattr(model, "cache", None)
    report = cache.report() if cache is not None else None
    if not report:
        return {"device": 0.0, "host": 0.0, "stub": True,
                "note": "this run had no weight cache; every byte came from storage every pass"}
    return {
        "device": report["device_hit_rate"],
        "host": report["host_hit_rate"],
        "stub": False,
        "acquires": report["hits_device"] + report["hits_host"] + report["misses"],
        "hits_device": report["hits_device"],
        "hits_host": report["hits_host"],
        "misses": report["misses"],
        "evicted_to_host": report["evicted_to_host"],
        "evicted_to_storage": report["evicted_to_storage"],
        "prefetches": report.get("prefetches", 0),
        "prefetch_hits": report.get("prefetch_hits", 0),
        "pinned": report["pinned"],
        "window": report["window"],
        "note": (f"window {report['window']} layers, {report['pinned']} pinned, "
                 f"{report['device_entries']} resident on the device"),
    }


def _skew_verdict(experts):
    """Say plainly whether this run's routing justifies keeping hot experts resident.

    The residency policy is a bet that some experts are read far more than others. If a model does
    not route that way the bet does not pay, and the honest thing is to print that rather than let
    a reader assume the numbers above are a win.
    """
    if not experts["selections"]:
        return "nothing routed: no expert popularity to exploit"
    gini = experts["gini"]
    if gini < 0.10:
        return ("routing is near-uniform, so expert residency has little to exploit here; "
                "randomly initialised weights route this way")
    if gini < 0.30:
        return "routing is mildly skewed: pinning helps, but the tail is still read often"
    return "routing is strongly skewed, which is what makes hot-expert residency pay"


def expert_cache(model):
    """What a mixture's experts did, or ``None`` for a dense model.

    The expert hit rate is reported apart from the overall one on purpose. Experts and dense layers
    have different access patterns and different replacement policies, so a single combined figure
    averages away the only number that says whether expert residency is working.

    The routing distribution is here because the whole approach rests on it. Keeping hot experts
    resident only pays if a mixture really does prefer some experts over others, and that is a
    property of the trained weights rather than of this engine -- so it is measured and printed
    rather than assumed. `gini` is 0 when every expert is chosen equally often and approaches 1 as
    the choices concentrate; `top_10pct_share` is printed beside what that decile would hold under
    uniform routing, so a flat distribution is visible at a glance rather than inferred.
    """
    reporter = getattr(model, "expert_report", None)
    return reporter() if callable(reporter) else None


def speculation(model):
    """What speculation did, or why it was not doing anything.

    Reported even when it is off, and with the draft's resident bytes alongside the acceptance
    rate, because the two halves of the trade have to be read together: the draft buys tokens per
    pass and pays for them in residency the weight cache would otherwise have had. A run that shows
    a healthy acceptance rate and a collapsed cache hit rate has lost.
    """
    decoder = getattr(model, "spec", None)
    if decoder is None:
        return {"enabled": False}
    stats = decoder.stats.to_dict()
    stats.update(enabled=True,
                 draft=model.draft.name,
                 draft_resident_bytes=model.draft.device_bytes(),
                 draft_forwards=model.draft.forwards,
                 lookahead_ceiling=decoder.max_lookahead,
                 lookahead_final=decoder.lookahead)
    return stats


def build_record(args, device, meters, marker, totals, device_tier_bytes, prompt_tokens,
                 new_tokens, wall_seconds, build_seconds, peak_device,
                 physical_before, physical_after, text, model, hardware):
    # The first mark closes prefill; everything after it is decode. Counters are snapshotted at
    # each mark, so the prefill/decode byte split falls out without a second run.
    marks = marker.marks
    if marks:
        prefill_snapshot = marks[0][1]
        decode_snapshot = diff_snapshots(totals, prefill_snapshot)
    else:
        prefill_snapshot = totals
        decode_snapshot = {key: 0 for key in totals}

    if len(marks) >= 2:
        decode_seconds = marks[-1][0] - marks[0][0]
        # Tokens rather than marks: a speculative pass is one mark and several tokens.
        decode_steps = marks[-1][2] - marks[0][2]
        prefill_seconds = wall_seconds - decode_seconds
    else:
        # A single generated token gives no decode interval to measure.
        decode_seconds = UNAVAILABLE
        decode_steps = 0
        prefill_seconds = wall_seconds

    per_token = (lambda value: value / new_tokens) if new_tokens else (lambda value: UNAVAILABLE)

    physical = UNAVAILABLE
    if physical_before is not None and physical_after is not None:
        physical = physical_after - physical_before

    record = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware_profile": hardware,
        "software": software_env(),
        "run": {
            "model": args.model,
            "prompt_tokens": prompt_tokens,
            "new_tokens": new_tokens,
            "max_vram_gb": args.max_vram_gb,
            "prefetching": not args.no_prefetch,
            "sync_phases": not args.no_sync_phases,
            "dtype": str(model.running_dtype),
            "draft_model": args.draft_model,
            "output_preview": text[:200],
        },
        "throughput": {
            "wall_seconds": wall_seconds,
            "model_build_seconds": build_seconds,
            "prefill_seconds": prefill_seconds,
            "prefill_tokens_per_second": (prompt_tokens / prefill_seconds
                                          if prefill_seconds and prefill_seconds > 0 else UNAVAILABLE),
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": (decode_steps / decode_seconds
                                         if decode_seconds and decode_seconds > 0 else UNAVAILABLE),
        },
        "bytes_per_token": {
            # Resident weights are re-read from device memory for every token, so the per-token
            # figure is the resident footprint itself.
            "device": device_tier_bytes if new_tokens else UNAVAILABLE,
            "host": per_token(totals["transfer_bytes"]),
            "storage": per_token(totals["storage_bytes"]),
        },
        "bytes_total": {
            "device_resident": device_tier_bytes,
            "host_transferred": totals["transfer_bytes"],
            "storage_read": totals["storage_bytes"],
            "storage_read_exact": meters.storage_bytes_exact,
            "storage_physical_read": physical,
            "prefill": {"host": prefill_snapshot["transfer_bytes"],
                        "storage": prefill_snapshot["storage_bytes"]},
            "decode": {"host": decode_snapshot["transfer_bytes"],
                       "storage": decode_snapshot["storage_bytes"]},
        },
        "cache_hit_rate": cache_hit_rate(model),
        "expert_cache": expert_cache(model),
        "speculation": speculation(model),
        # The critical path: these four are sequential and sum toward wall time. The worker-thread
        # read time is reported alongside them but deliberately kept out of the sum, because with
        # prefetching on it runs underneath compute and adding it would double-count.
        "phases_seconds": {
            "storage_wait": totals["storage_wait_seconds"],
            "host_to_device": totals["transfer_seconds"],
            "dequant": totals["dequant_seconds"],
            "compute": totals["compute_seconds"],
            "evict": totals["evict_seconds"],
            "unattributed": wall_seconds - (totals["storage_wait_seconds"]
                                            + totals["transfer_seconds"]
                                            + totals["dequant_seconds"]
                                            + totals["compute_seconds"]
                                            + totals["evict_seconds"]),
            "storage_read_overlapped": totals["storage_seconds"],
            "note": ("storage_wait is what the forward actually blocked for; "
                     "storage_read_overlapped is the loader's own read time, which runs on the "
                     "prefetch worker and hides behind compute"
                     if not args.no_prefetch else
                     "prefetching disabled, so storage_wait and storage_read_overlapped describe "
                     "the same inline reads"),
        },
        "counts": {
            "storage_reads": totals["storage_reads"],
            "device_placements": totals["transfers"],
        },
        "memory": {
            "peak_device_bytes": peak_device,
            "peak_host_rss_bytes": peak_host_rss_bytes(),
        },
    }
    return record


# ---------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------

def human_bytes(value):
    if value is None:
        return "unavailable"
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < step or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= step
    return f"{value:.1f} TB"


def fmt(value, suffix="", digits=2):
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def print_report(record):
    profile = record["hardware_profile"]
    run_info = record["run"]
    tp = record["throughput"]
    bpt = record["bytes_per_token"]
    totals = record["bytes_total"]
    phases = record["phases_seconds"]

    print()
    print("=" * 78)
    print("  RocketLLM streaming benchmark")
    print("=" * 78)
    print(f"  model         {run_info['model']}")
    print(f"  tokens        {run_info['prompt_tokens']} prompt -> {run_info['new_tokens']} generated")
    print(f"  dtype         {run_info['dtype']}")
    identity = hardware_identity(profile)
    print(f"  device        {identity['device']}  [{identity['backend']}]  "
          f"cc {fmt(identity['compute_capability'])}")
    print(f"  device memory {human_bytes(identity['device_total_bytes'])}   "
          f"host RAM {human_bytes(identity['host_total_bytes'])}")
    cap = "none requested" if run_info["max_vram_gb"] is None else f"{run_info['max_vram_gb']}GB"
    dtypes = profile.get("dtypes") or {}
    print(f"  bf16          {fmt(dtypes.get('bf16'))}    "
          f"pinned {fmt(profile.get('pinned_memory'))}    "
          f"vram cap {cap}")
    print(f"  profile key   {profile.get('profile_key')}")

    # The knobs the profile chose for this run. A later run that derives different ones is not
    # measuring the same configuration, so they belong next to the numbers they produced.
    derived = profile.get("derived") or {}
    if derived:
        interesting = ("reserve_bytes", "host_cache_bytes", "io_workers",
                       "compute_dtype", "kv_dtype", "quant_compute_path")
        parts = []
        for name in interesting:
            entry = derived.get(name)
            if not entry:
                continue
            value = entry["value"]
            parts.append(f"{name}={human_bytes(value) if name.endswith('_bytes') else value}")
        print(f"  knobs         {'  '.join(parts)}")
    for warning in (profile.get("warnings") or []):
        print(f"  ! {warning}")

    print()
    print("  BYTES PER TOKEN BY TIER  (the primary metric)")
    print(f"    device  (resident, no move)   {human_bytes(bpt['device'])}")
    print(f"    host    (over the link)       {human_bytes(bpt['host'])}")
    print(f"    storage (read from disk)      {human_bytes(bpt['storage'])}")
    if not totals["storage_read_exact"]:
        print("    note: storage bytes summed from tensors, not file sizes, on this layout")
    if totals["storage_physical_read"] is not None:
        print(f"    physical disk reads (total)   {human_bytes(totals['storage_physical_read'])}"
              "  [rest served by page cache]")
    else:
        print("    physical disk reads           unavailable on this platform")

    print()
    print("  THROUGHPUT")
    print(f"    prefill   {fmt(tp['prefill_seconds'], 's')}   "
          f"{fmt(tp['prefill_tokens_per_second'], ' tok/s')}")
    print(f"    decode    {fmt(tp['decode_seconds'], 's')}   "
          f"{fmt(tp['decode_tokens_per_second'], ' tok/s')}")
    print(f"    wall      {fmt(tp['wall_seconds'], 's')}   "
          f"(model build {fmt(tp['model_build_seconds'], 's')}, excluded)")

    print()
    print("  WALL-CLOCK PHASES  (critical path, sums toward wall time)")
    for label, key in (("storage wait", "storage_wait"), ("host -> device", "host_to_device"),
                       ("dequant", "dequant"), ("compute", "compute"), ("evict", "evict"),
                       ("unattributed", "unattributed")):
        share = ""
        if phases[key] is not None and tp["wall_seconds"]:
            share = f"   {phases[key] / tp['wall_seconds']:6.1%}"
        print(f"    {label:<16}{fmt(phases[key], 's'):>10}{share}")
    print(f"    {'':<16}{'':>10}")
    print(f"    storage read on the prefetch worker (overlapped, not in the sum above): "
          f"{fmt(phases['storage_read_overlapped'], 's')}")
    print(f"    {phases['note']}")

    print()
    print("  CACHE HIT RATE")
    print(f"    device {record['cache_hit_rate']['device']:.0%}   "
          f"host {record['cache_hit_rate']['host']:.0%}   "
          f"({record['cache_hit_rate']['note']})")

    experts = record.get("expert_cache")
    if experts:
        print()
        print("  EXPERT CACHE")
        print(f"    hit rate  {experts['hit_rate']:.0%}   "
              f"(device {experts['hits_device']}, host {experts['hits_host']}, "
              f"miss {experts['misses']} of {experts['acquires']} expert acquires)")
        print(f"    residency {experts['pinned_experts']} experts pinned, "
              f"{experts['pinned_shared']}/{experts['shared_modules']} shared modules pinned "
              f"({human_bytes(experts['shared_bytes'])}), {experts['replans']} re-rankings")
        print(f"    routing   {experts['distinct_per_visit']:.2f} distinct experts per layer visit, "
              f"{experts['touched']}/{experts['experts']} ever routed to, "
              f"{experts['firings']} router firings")
        print(f"    parallel  {experts['prefetched']} expert reads issued as a batch on "
              f"{experts['prefetch_calls']} router firings")
        print(f"    skew      gini {experts['gini']:.2f}   "
              f"hottest expert {experts['top_1_share']:.1%} of selections   "
              f"hottest decile {experts['top_10pct_share']:.1%} "
              f"(uniform would be {experts['uniform_share']:.1%})")
        print(f"    {_skew_verdict(experts)}")

    spec = record.get("speculation") or {}
    if spec.get("enabled"):
        print()
        print("  SPECULATION")
        print(f"    draft     {spec['draft']} resident "
              f"({human_bytes(spec['draft_resident_bytes'])}), "
              f"{spec['draft_forwards']} draft forwards")
        print(f"    accepted  {spec['acceptance_rate']:.0%} of {spec['proposed']} proposals "
              f"(mean lookahead {spec['mean_lookahead']:.1f}, ceiling "
              f"{spec['lookahead_ceiling']})")
        print(f"    yield     {spec['tokens_per_pass']:.2f} tokens per verification pass "
              f"({spec['emitted']} tokens in {spec['passes']} passes)")
        # The number that decides whether any of this was worth it. One token per pass is what
        # ordinary decoding produces, so at 1.0 speculation bought the draft's memory and its
        # forwards and returned nothing for them.
        verdict = ("BELOW plain decoding -- the draft cost passes and residency for nothing"
                   if spec['tokens_per_pass'] <= 1.0 else
                   f"{spec['tokens_per_pass']:.2f}x the passes plain decoding would have needed")
        print(f"    verdict   {verdict}")
        print(f"    draft time {fmt(spec['draft_seconds'], 's')} against "
              f"{fmt(spec['target_seconds'], 's')} of verification passes")
    elif record.get("run", {}).get("draft_model"):
        print()
        print("  SPECULATION")
        print("    off for this run; the draft model was configured but not enabled")

    print()
    print("  PEAK MEMORY")
    print(f"    device    {human_bytes(record['memory']['peak_device_bytes'])}")
    print(f"    host rss  {human_bytes(record['memory']['peak_host_rss_bytes'])}")

    print()
    print("  TOTALS")
    print(f"    storage read {human_bytes(totals['storage_read'])} in "
          f"{record['counts']['storage_reads']} reads    "
          f"host->device {human_bytes(totals['host_transferred'])} in "
          f"{record['counts']['device_placements']} placements")
    print(f"    prefill: {human_bytes(totals['prefill']['storage'])} storage / "
          f"{human_bytes(totals['prefill']['host'])} host      "
          f"decode: {human_bytes(totals['decode']['storage'])} storage / "
          f"{human_bytes(totals['decode']['host'])} host")
    print("=" * 78)


# metric path, label, unit, whether a rise is an improvement
COMPARED_METRICS = [
    ("bytes_per_token.storage", "storage bytes/token", "bytes", False),
    ("bytes_per_token.host", "host bytes/token", "bytes", False),
    ("bytes_per_token.device", "device bytes/token", "bytes", True),
    ("throughput.decode_tokens_per_second", "decode tok/s", "rate", True),
    ("throughput.prefill_tokens_per_second", "prefill tok/s", "rate", True),
    ("throughput.wall_seconds", "wall seconds", "seconds", False),
    ("phases_seconds.storage_wait", "phase: storage wait", "seconds", False),
    ("phases_seconds.host_to_device", "phase: host->device", "seconds", False),
    ("phases_seconds.dequant", "phase: dequant", "seconds", False),
    ("phases_seconds.compute", "phase: compute", "seconds", False),
    ("phases_seconds.evict", "phase: evict", "seconds", False),
    ("memory.peak_device_bytes", "peak device memory", "bytes", False),
    ("memory.peak_host_rss_bytes", "peak host rss", "bytes", False),
    ("cache_hit_rate.device", "cache hit rate (device)", "ratio", True),
    # Absent on a dense model, which the differ reports as unavailable rather than as zero.
    ("expert_cache.hit_rate", "expert hit rate", "ratio", True),
    ("expert_cache.distinct_per_visit", "experts per layer visit", "rate", False),
    ("expert_cache.gini", "routing skew (gini)", "ratio", True),
    # Absent unless speculation ran. Comparing a speculative run against a plain one is how the
    # whole trade is read: tokens per pass on one side, cache hit rate and peak memory on the
    # other, because the draft's residency comes out of the weight cache.
    ("speculation.acceptance_rate", "draft acceptance rate", "ratio", True),
    ("speculation.tokens_per_pass", "tokens per pass", "rate", True),
    ("speculation.draft_resident_bytes", "draft resident bytes", "bytes", False),
]


def dig(record, path):
    node = record
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return UNAVAILABLE
        node = node[part]
    return node


def print_comparison(current, previous, forced):
    old_profile = previous.get("hardware_profile", {})
    new_profile = current["hardware_profile"]
    old_identity, new_identity = hardware_identity(old_profile), hardware_identity(new_profile)
    if old_identity != new_identity:
        differing = [key for key in old_identity if old_identity[key] != new_identity[key]]
        message = (f"hardware profiles differ ({', '.join(differing) or 'unknown fields'}): "
                   f"{old_identity.get('device')} vs {new_identity.get('device')}")
        if not forced:
            print(f"\nREFUSING TO COMPARE: {message}.\n"
                  f"Bytes per token is comparable across machines; timings are not. "
                  f"Re-run with --force to compare anyway.")
            return False
        print(f"\nWARNING: {message}. Comparing anyway because --force was given. "
              f"Treat every timing below as meaningless; only the byte counts carry over.")

    old_software = previous.get("software", {})
    if old_software != current["software"]:
        print(f"\nnote: software differs -- was {old_software}, now {current['software']}")

    old_run, new_run = previous.get("run", {}), current["run"]
    for key in ("model", "prompt_tokens", "new_tokens", "prefetching", "sync_phases"):
        if old_run.get(key) != new_run.get(key):
            print(f"note: run config differs on {key}: {old_run.get(key)} -> {new_run.get(key)}")

    print()
    print("=" * 78)
    print("  DELTA vs previous")
    print("=" * 78)
    print(f"  {'metric':<28}{'previous':>14}{'current':>14}{'change':>20}")
    print("  " + "-" * 74)
    for path, label, unit, higher_is_better in COMPARED_METRICS:
        old_value, new_value = dig(previous, path), dig(current, path)
        render = human_bytes if unit == "bytes" else (lambda v: fmt(v, "", 2))
        if old_value is None or new_value is None:
            print(f"  {label:<28}{render(old_value):>14}{render(new_value):>14}{'--':>20}")
            continue
        if old_value == 0:
            change = "n/a (was 0)"
        elif render(old_value) == render(new_value):
            # Two values that print the same differ only below display resolution; quoting a
            # percentage there would dress up noise as a result.
            change = "same"
        else:
            pct = (new_value - old_value) / abs(old_value) * 100.0
            better = (pct > 0) == higher_is_better
            direction = "better" if abs(pct) > 0.5 and better else (
                "worse" if abs(pct) > 0.5 else "same")
            change = f"{pct:+.1f}%  {direction}"
        print(f"  {label:<28}{render(old_value):>14}{render(new_value):>14}{change:>20}")
    print("=" * 78)
    return True


def write_record(record, explicit_path=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if explicit_path:
        path = Path(explicit_path)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        slug = record["run"]["model"].replace("/", "_").replace(":", "_")
        key = record["hardware_profile"]["profile_key"]
        path = RESULTS_DIR / f"{stamp}_{slug}_{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-vram-gb", type=float, default=None,
                        help="emulate a smaller device (CUDA allocator only)")
    parser.add_argument("--device", default=None,
                        help="override the auto-detected backend, e.g. cpu or cuda:1")
    parser.add_argument("--json", action="store_true",
                        help=f"write the result record under {RESULTS_DIR.name}/")
    parser.add_argument("--out", default=None, help="explicit path for the result record")
    parser.add_argument("--compare-to", default=None, help="a previous result record to diff against")
    parser.add_argument("--vram-reserve", type=int, default=None,
                        help="bytes of device memory to hold back, shrinking the weight cache. "
                             "--max-vram-gb caps the allocator but not the measured budget, so this "
                             "is the lever for benchmarking a model that does not fit")
    parser.add_argument("--no-expert-residency", action="store_true",
                        help="stop keeping hot experts resident, to measure the policy against its "
                             "own absence on one build")
    parser.add_argument("--host-cache-gb", type=float, default=None,
                        help="gigabytes the host tier may hold. Set to 0 to measure the "
                             "storage-bound regime, where evictions fall all the way to disk")
    parser.add_argument("--force", action="store_true",
                        help="compare across different hardware profiles anyway")
    parser.add_argument("--no-prefetch", action="store_true",
                        help="disable the prefetch worker so phase times stop overlapping")
    parser.add_argument("--no-sync-phases", action="store_true",
                        help="skip device synchronization; truer wall time, vaguer phase split")
    parser.add_argument("--draft-model", default=None,
                        help="a small model sharing this one's tokenizer, kept resident to propose "
                             "tokens that one streaming pass then verifies. Compare a run with it "
                             "against one without: the draft buys tokens per pass and pays for "
                             "them in the residency the weight cache loses")
    parser.add_argument("--speculative", default="auto", choices=("auto", "on", "off"),
                        help="auto takes the profile's measured recommendation, which on a machine "
                             "where the model already fits is no")
    parser.add_argument("--reprofile", action="store_true",
                        help="re-measure the hardware profile instead of using the cached one")
    parser.add_argument("--conversation", type=int, default=0, metavar="TURNS",
                        help="instead of one generation, replay a multi-turn conversation with the "
                             "prefix cache on and off, and report the prefill time of each turn")
    parser.add_argument("--conversation-tokens", type=int, default=24,
                        help="tokens generated per conversation turn (default: 24)")
    parser.add_argument("--conversation-filler", type=int, default=50,
                        help="how many times the opening paragraph is repeated, which is how the "
                             "first prompt is made long enough for prefill to dominate. The "
                             "default lands around 1500 tokens; raise it on a model with room")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"benchmark device: {device} ({device_name(device)})")
    if device.type == "cpu":
        print("note: CPU is a correctness target, not a performance one. Device-tier and "
              "peak-device metrics will report as unavailable.")

    cap_vram(args.max_vram_gb)

    if args.conversation:
        run_conversation(args, device)
        return

    record = run(args, device)
    print_report(record)

    if args.json or args.out:
        path = write_record(record, args.out)
        print(f"\nwrote {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")

    if args.compare_to:
        previous = json.loads(Path(args.compare_to).read_text(encoding="utf-8"))
        if not print_comparison(record, previous, args.force):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
