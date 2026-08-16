"""``rocketllm doctor``: one page describing the machine, for a bug report to carry.

RocketLLM has no reference machine, so "it is slow" and "it produced nonsense" are not reports
anybody can act on. What they become actionable against is the hardware the run happened on and the
decisions that hardware forced: which dtype the device can really do, whether a fused kernel was
found, which tier ends up serving the weights, and how fast that tier actually measured. All of that
is already known to :mod:`rocketllm.hw.profile` and :mod:`rocketllm.hw.caps`; this module is the one
place that puts it in front of a person.

Four things are printed that nothing else prints together:

* the **capability decision table** -- every gate, the answer, and the fallback taken when the
  answer is no. A user seeing "bf16: no -> fp16, output may be silently wrong on a deep model" has
  been told the thing that explains their bug.
* the **optional package inventory** -- what is installed, what each one would unlock, and what
  happens without it. Half of the reports this project will get are a missing kernel package.
* the **storage verdict** -- the measured read bandwidth of the filesystem the weights are on, and a
  loud warning when it is rotational or merely slow, because on a streaming engine that single
  number can dominate everything else by orders of magnitude.
* a **projected per-token cost** for a model of a given size, computed from the measured bandwidths
  through the performance model the whole engine is designed around.

Nothing here measures anything itself. It asks the profile, and where the profile could not measure
something it says so rather than substituting a number -- an invented bandwidth in a bug report is
worse than an absent one, because someone will act on it.
"""
import dataclasses
import json
from pathlib import Path

from . import caps
from .profile import DEFAULT_POLICY, HardwareProfile, _bw, _bytes, _fmt

#: Checkpoint file extensions worth counting toward a model's on-disk size.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth")


# -------------------------------------------------------------------------------------------------
# optional packages
# -------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class OptionalPackage:
    """One package RocketLLM can use but never requires.

    `without` is the whole point of the entry. Anyone can run ``pip list``; what they cannot do is
    tell which of the missing names actually costs them anything on their hardware, and which is a
    kernel for a card they do not own.
    """

    module: str
    #: What to type to get it, or None where it is not something this project ships an extra for.
    install: object
    unlocks: str
    without: str

    def status(self):
        return {
            "module": self.module,
            "present": caps._importable(self.module),
            "install": self.install,
            "unlocks": self.unlocks,
            "without": self.without,
        }


#: Every package RocketLLM will use if it is there. Kept as data rather than as scattered import
#: guards so that this list, the doctor's output and the optional-import test cannot drift apart.
OPTIONAL_PACKAGES = (
    OptionalPackage(
        "fastapi", "pip install 'rocketllm[server]'",
        "the OpenAI-compatible HTTP server (rocketllm serve)",
        "generate() from Python still works; only the server is unavailable"),
    OptionalPackage(
        "uvicorn", "pip install 'rocketllm[server]'",
        "the ASGI server rocketllm serve runs on",
        "generate() from Python still works; only the server is unavailable"),
    OptionalPackage(
        "pydantic", "pip install 'rocketllm[server]'",
        "request and response validation for the OpenAI schemas",
        "generate() from Python still works; only the server is unavailable"),
    OptionalPackage(
        "PIL", "pip install 'rocketllm[vision]'",
        "decoding the images in a multimodal chat request",
        "text generation is unaffected; a request carrying an image is refused with this hint"),
    OptionalPackage(
        "torchvision", "pip install 'rocketllm[vision]'",
        "the processors of video-capable checkpoints, and transformers' faster image processors",
        "a checkpoint whose processor declares a video processor cannot build one at all "
        "(Qwen2.5-VL is one), so it serves text only; others just preprocess more slowly"),
    OptionalPackage(
        "compressed_tensors", "pip install 'rocketllm[quant]'",
        "reading compressed-tensors W4A16 and MXFP4 checkpoints",
        "those checkpoints cannot be loaded; every other format still can"),
    OptionalPackage(
        "bitsandbytes", "pip install 'rocketllm[quant]'",
        "reading bitsandbytes-prequantized checkpoints, and its fused 4-bit matmul",
        "those checkpoints cannot be loaded; other 4-bit formats dequantize into scratch"),
    OptionalPackage(
        "triton", None,
        "the hand-written dequant kernels",
        "dequantization falls back to the PyTorch implementation, which is slower and identical"),
    OptionalPackage(
        "gptqmodel", None,
        "a fused 4-bit matmul for GPTQ checkpoints",
        "GPTQ weights are expanded into scratch before the matmul"),
    OptionalPackage(
        "awq_ext", None,
        "a fused 4-bit matmul for AWQ checkpoints",
        "AWQ weights are expanded into scratch before the matmul"),
    OptionalPackage(
        "exllamav2", None,
        "an alternative fused 4-bit matmul",
        "packed weights are expanded into scratch before the matmul"),
    OptionalPackage(
        "marlin_kernels", None,
        "the Marlin fused 4-bit matmul",
        "packed weights are expanded into scratch before the matmul"),
    OptionalPackage(
        "hqq", None,
        "transformers' HQQ quantized KV cache, as a reference for the built-in int4 cache",
        "kv_cache='hqq' is unavailable; 'auto', 'fp16' and 'int4' are unaffected"),
    OptionalPackage(
        "optimum", None,
        "transformers' quanto quantized KV cache, as a reference for the built-in int4 cache",
        "kv_cache='quanto' is unavailable; 'auto', 'fp16' and 'int4' are unaffected"),
    OptionalPackage(
        "mlx", "pip install 'rocketllm[mlx]'",
        "the Apple Silicon MLX path",
        "Apple Silicon runs through the MPS backend instead"),
    OptionalPackage(
        "psutil", "pip install 'rocketllm[mlx]'",
        "host memory reporting on the MLX path",
        "the portable memory probe is used instead; nothing is lost off the MLX path"),
)


def package_inventory():
    """Every optional package, whether it imports here, and what its absence costs."""
    return [package.status() for package in OPTIONAL_PACKAGES]


# -------------------------------------------------------------------------------------------------
# capability decision table
# -------------------------------------------------------------------------------------------------

def capability_rows(device_caps=None):
    """Every capability gate: the answer, how it was decided, and the fallback if the answer is no.

    Deliberately built from a live :class:`~rocketllm.hw.caps.DeviceCaps` rather than from the
    profile's cached copy. The profile is keyed by a hardware fingerprint and replayed, so a driver
    change that alters an answer would be invisible in it until something invalidates the cache;
    a bug report needs today's answer.
    """
    dev = device_caps if device_caps is not None else caps.get_caps(announce=False)
    plan = dev.fused_4bit_plan()
    providers = dev.fused_4bit_providers()
    usable = sorted(name for name, ok in providers.items() if ok)

    rows = [
        {
            "capability": "bf16",
            "answer": dev.supports_bf16,
            "decided_by": "a real bfloat16 matmul was attempted on this device",
            "fallback": ("fp16, whose range overflows to inf/NaN on very deep models and corrupts "
                         "output SILENTLY rather than raising -- suspect this first if results "
                         "look wrong"),
        },
        {
            "capability": "fp16",
            "answer": dev.supports_fp16,
            "decided_by": "a real float16 matmul was attempted on this device",
            "fallback": "fp32: correct, and twice the bytes moved per token",
        },
        {
            "capability": "fp8",
            "answer": dev.supports_fp8,
            "decided_by": "torch._scaled_mm was called on fp8 operands",
            "fallback": "fp8 checkpoints are read, then computed in the compute dtype",
        },
        {
            "capability": "native fp4",
            "answer": dev.supports_fp4,
            "decided_by": "an fp4 tensor was allocated on this device",
            "fallback": "4-bit formats are dequantized before the matmul, which is the usual path",
        },
        {
            "capability": "pinned host memory",
            "answer": dev.can_pin_memory,
            "decided_by": "a page-locked host buffer was allocated",
            "fallback": "pageable staging buffers: the same bytes, transferred more slowly",
        },
        {
            "capability": "async copy streams",
            "answer": dev.has_async_streams,
            "decided_by": "a copy stream was created on this backend",
            "fallback": "the synchronous transfer path; reads do not overlap compute",
        },
        {
            "capability": "fused 4-bit matmul",
            "answer": plan.fused,
            "decided_by": (f"kernel packages that import and can run here: "
                           f"{', '.join(usable) if usable else 'none'}"),
            "fallback": "packed weights are expanded into a reusable scratch buffer first",
        },
        {
            "capability": "triton dequant kernels",
            "answer": caps.has_triton(),
            "decided_by": "the triton package was imported",
            "fallback": "the PyTorch dequant implementation: same numbers, slower",
        },
    ]
    for row in rows:
        # A capability that is present has not taken its fallback, and saying so keeps the table
        # readable at a glance -- the eye should land only on the rows that cost something.
        row["taken"] = None if row["answer"] else row["fallback"]
    return rows


# -------------------------------------------------------------------------------------------------
# storage
# -------------------------------------------------------------------------------------------------

def storage_health(profile, policy=DEFAULT_POLICY):
    """The weights-path read bandwidth, and how alarmed to be about it.

    Two separate alarms, because they are separate facts. Rotational is what the OS says the device
    is; slow is what the probe measured. A device can be one without the other -- a fast disk behind
    a saturated USB link measures slow and is not rotational -- and a report that conflated them
    would send someone looking at the wrong thing.
    """
    storage = profile.storage or {}
    best = storage.get("best_bytes_per_s")
    verdict = {
        "path": storage.get("path"),
        "bytes_per_s": best,
        "queue_depth_1_bytes_per_s": storage.get("queue_depth_1_bytes_per_s"),
        "saturating_concurrency": storage.get("saturating_concurrency"),
        "rotational": storage.get("rotational"),
        "probed_real_shards": bool(storage.get("probed_real_shards")),
        "synthetic_probe": bool(storage.get("synthetic_probe")),
        "page_cache_influence": storage.get("page_cache_influence"),
        "error": storage.get("error"),
        "alarms": [],
    }
    if storage.get("rotational"):
        verdict["alarms"].append(
            "THE WEIGHTS ARE ON A DEVICE THE OS REPORTS AS ROTATIONAL. A streaming engine reads "
            "every unresident weight on every token, and a seek-bound device turns that into the "
            "single dominant cost of the run -- expect per-token times measured in seconds, not "
            "milliseconds. Move the weights to an SSD before drawing any conclusion about speed.")
    if best is not None and best < policy.slow_storage_bytes_per_s:
        verdict["alarms"].append(
            f"STORAGE READ BANDWIDTH MEASURED AT {best / 1e6:.0f} MB/s, which is slow enough to "
            f"dominate everything else this engine does. Every gigabyte of non-resident weight "
            f"costs about {1e9 / best:.1f}s per token.")
    if storage.get("synthetic_probe"):
        verdict["alarms"].append(
            "no checkpoint shards were found at the weights path, so the number above came from a "
            "temporary file written on the same filesystem. It describes the right device but not "
            "the real read pattern.")
    if storage.get("error"):
        verdict["alarms"].append(f"storage was not probed: {storage['error']}. Pass --weights-path "
                                 f"pointing at the model to get this measured.")
    return verdict


# -------------------------------------------------------------------------------------------------
# the per-token projection
# -------------------------------------------------------------------------------------------------

def checkpoint_bytes(path):
    """Total weight bytes on disk under `path`, or None if there is nothing to measure.

    Measured rather than derived from a parameter count, because what the engine moves is what the
    file holds: a 4-bit checkpoint of a 70B model is not 70B times anything the caller would guess.
    """
    if not path:
        return None
    root = Path(path)
    if root.is_file():
        try:
            return int(root.stat().st_size)
        except OSError:
            return None
    if not root.is_dir():
        return None
    total = 0
    found = False
    for candidate in root.rglob("*"):
        if candidate.suffix.lower() not in _WEIGHT_SUFFIXES:
            continue
        try:
            total += candidate.stat().st_size
        except OSError:
            continue
        found = True
    return total if found else None


def _tier_seconds(nbytes, bandwidth):
    if not nbytes:
        return 0.0
    if not bandwidth:
        return None
    return nbytes / bandwidth


def per_token_projection(profile, model_bytes):
    """Where a model of this size would live, and what one token would therefore cost.

    This is the project's performance model applied to measured numbers: time per token is the sum
    over tiers of the bytes that tier has to serve, divided by what that tier measured at. The
    placement is the engine's own -- the pin budget is what is left of usable device memory once the
    prefetch window is committed, the host cache takes what will not fit there, and storage serves
    the rest on every single token.

    The storage tier is charged at ``min(read bandwidth, host->device bandwidth)``: a byte read from
    disk still has to cross the link to reach the device, the loader overlaps those two stages, and
    a pipeline runs at the speed of its slower stage. Charging the read alone would understate the
    cost on a machine with a fast NVMe behind a slow link.

    Any tier whose bandwidth was never measured contributes ``None``, and the total says it is
    incomplete. Substituting a plausible number here would produce a projection someone acts on.
    """
    model_bytes = int(model_bytes or 0)
    derived = profile.derived or {}

    def knob(name):
        derivation = derived.get(name)
        return int(derivation.value) if derivation is not None and derivation.value else 0

    usable = knob("usable_device_bytes")
    window = knob("window_budget_bytes")
    pin_budget = max(0, usable - window)
    host_budget = knob("host_cache_bytes")

    device_bytes = min(model_bytes, pin_budget)
    remainder = model_bytes - device_bytes
    host_bytes = min(remainder, host_budget)
    storage_bytes = remainder - host_bytes

    link = (profile.host_to_device_pinned_bandwidth
            or profile.host_to_device_pageable_bandwidth)
    read = (profile.storage or {}).get("best_bytes_per_s")
    # On a backend with no separate device pool the "link" is a copy within one memory, and the
    # profile records nothing for it. Falling back to the device bandwidth there is not a guess:
    # it is the same memory the device tier was just measured on.
    if link is None and profile.device_type == "cpu":
        link = profile.device_memory_bandwidth
    # The link only ever *bounds* the storage tier. Where storage itself was never measured the
    # answer is "unknown", not "the link speed" -- substituting the faster of the two stages for the
    # one nobody timed would report a storage-bound machine as a fast one.
    storage_effective = None if not read else (min(read, link) if link else read)

    tiers = [
        {"tier": "device", "bytes": device_bytes, "bandwidth": profile.device_memory_bandwidth,
         "note": "resident weights, read by the matmul at device memory bandwidth"},
        {"tier": "host", "bytes": host_bytes, "bandwidth": link,
         "note": "held in the host cache, transferred over the link on every token"},
        {"tier": "storage", "bytes": storage_bytes, "bandwidth": storage_effective,
         "note": "re-read from the filesystem on every token, at min(read, link)"},
    ]
    total = 0.0
    complete = True
    for tier in tiers:
        tier["seconds"] = _tier_seconds(tier["bytes"], tier["bandwidth"])
        if tier["seconds"] is None:
            complete = False
        else:
            total += tier["seconds"]
        tier["share"] = (tier["bytes"] / model_bytes) if model_bytes else 0.0

    return {
        "model_bytes": model_bytes,
        "tiers": tiers,
        "seconds_per_token": total if model_bytes else 0.0,
        "tokens_per_second": (1.0 / total) if total > 0 else None,
        "complete": complete,
        "fits_resident": storage_bytes == 0 and host_bytes == 0,
        "placement": {
            "usable_device_bytes": usable,
            "window_budget_bytes": window,
            "pin_budget_bytes": pin_budget,
            "host_cache_bytes": host_budget,
        },
    }


def model_bytes_from(model=None, model_bytes=None, params=None, weight_bits=None):
    """Settle how large the model to project for is, from whichever input was given.

    Three ways in, in decreasing order of how much they are worth trusting: a real checkpoint on
    disk, an explicit byte count, and a parameter count times a bit width. The last is the one a
    user reaches for and the loosest, so what it produced is reported alongside the answer.
    """
    measured = checkpoint_bytes(model)
    if measured:
        return measured, f"measured from the checkpoint files under {model}"
    if model_bytes:
        return int(model_bytes), "given as a byte count"
    if params:
        bits = int(weight_bits or 16)
        return int(params * bits / 8), (f"{params / 1e9:.0f}B parameters at {bits} bits per weight "
                                        f"(the checkpoint's real size is what matters; pass "
                                        f"--model to measure it)")
    return None, None


# -------------------------------------------------------------------------------------------------
# the report
# -------------------------------------------------------------------------------------------------

def collect(weights_path=None, device=None, reprofile=False, model=None, model_bytes=None,
            params=None, weight_bits=None, storage_budget_seconds=3.0):
    """Everything the doctor knows, as data. `report` renders this; --json emits it."""
    # A model path is also a weights path: someone asking about a checkpoint wants the storage
    # number for the filesystem that checkpoint is on, not for whatever else was passed.
    probe_path = weights_path or model
    profile = HardwareProfile.load_or_probe(weights_path=probe_path, device=device,
                                            reprofile=reprofile,
                                            storage_budget_seconds=storage_budget_seconds)
    device_caps = caps.get_caps(device, announce=False)
    size, size_source = model_bytes_from(model=model, model_bytes=model_bytes, params=params,
                                         weight_bits=weight_bits)

    from ..quant import decision_table

    return {
        "profile": profile,
        "capabilities": capability_rows(device_caps),
        "packages": package_inventory(),
        "storage": storage_health(profile),
        "quant_formats": decision_table(caps=device_caps),
        "model_bytes": size,
        "model_bytes_source": size_source,
        "projection": per_token_projection(profile, size) if size else None,
    }


def to_dict(collected):
    """The same content as JSON, for pasting into an issue verbatim."""
    out = dict(collected)
    out["profile"] = collected["profile"].to_dict()
    return out


def _seconds(value):
    if value is None:
        return "unavailable"
    if value >= 1.0:
        return f"{value:.2f}s"
    return f"{value * 1000:.1f}ms"


def report(collected):
    """The whole thing as text. This is what a bug report is asked to paste."""
    profile = collected["profile"]
    lines = []
    add = lines.append
    rule = "=" * 78

    add(profile.describe())
    add("")
    add(rule)
    add("  CAPABILITY DECISIONS  (queried on this device, never inferred from its name)")
    add(rule)
    for row in collected["capabilities"]:
        answer = "yes" if row["answer"] else "NO"
        add(f"    {row['capability']:<24}{answer}")
        add(f"      decided by: {row['decided_by']}")
        if row["taken"]:
            add(f"      falls back to: {row['taken']}")
    add("")

    add(rule)
    add("  QUANTIZED CHECKPOINT FORMATS  (what each would do on this machine)")
    add(rule)
    for row in collected["quant_formats"]:
        add(f"    {row['format']:<24}{row['path']}")
        add(f"      {row['reason']}")
    add("")

    add(rule)
    add("  OPTIONAL PACKAGES")
    add(rule)
    for package in collected["packages"]:
        mark = "installed" if package["present"] else "MISSING  "
        add(f"    {mark}  {package['module']:<20}{package['unlocks']}")
        if not package["present"]:
            add(f"                without it: {package['without']}")
            if package["install"]:
                add(f"                install:    {package['install']}")
    add("")

    storage = collected["storage"]
    add(rule)
    add("  WEIGHT STORAGE")
    add(rule)
    add(f"    path              {_fmt(storage['path'])}")
    add(f"    read bandwidth    {_bw(storage['bytes_per_s'])}"
        f"  (queue depth 1: {_bw(storage['queue_depth_1_bytes_per_s'])})")
    add(f"    saturates at      {_fmt(storage['saturating_concurrency'])} concurrent readers")
    add(f"    rotational        {_fmt(storage['rotational'])}")
    add(f"    page cache        {_fmt(storage['page_cache_influence'])}")
    for alarm in storage["alarms"]:
        add("")
        add(f"    !! {alarm}")
    add("")

    projection = collected["projection"]
    add(rule)
    add("  PROJECTED COST PER TOKEN")
    add(rule)
    if projection is None:
        add("    No model size given, so there is nothing to project. Pass one of:")
        add("      --model PATH        measure a real checkpoint on disk (best)")
        add("      --model-bytes 40GB  a byte count you already know")
        add("      --model-size 70B --weight-bits 4")
    else:
        add(f"    model size        {_bytes(projection['model_bytes'])}"
            f"  ({collected['model_bytes_source']})")
        placement = projection["placement"]
        add(f"    device budget     {_bytes(placement['pin_budget_bytes'])} for resident weights "
            f"({_bytes(placement['usable_device_bytes'])} usable "
            f"- {_bytes(placement['window_budget_bytes'])} prefetch window)")
        add(f"    host cache        {_bytes(placement['host_cache_bytes'])}")
        add("")
        add(f"    {'tier':<10}{'bytes/token':>14}{'share':>9}{'bandwidth':>14}"
            f"{'time/token':>14}")
        for tier in projection["tiers"]:
            add(f"    {tier['tier']:<10}{_bytes(tier['bytes']):>14}"
                f"{tier['share'] * 100:>8.0f}%{_bw(tier['bandwidth']):>14}"
                f"{_seconds(tier['seconds']):>14}")
        add("")
        if projection["complete"]:
            rate = projection["tokens_per_second"]
            rendered = f"    projected         {_seconds(projection['seconds_per_token'])} per token"
            add(rendered + (f"  ({rate:.2f} tok/s)" if rate else ""))
        else:
            add(f"    projected         at least {_seconds(projection['seconds_per_token'])} per "
                f"token -- INCOMPLETE: a tier's bandwidth was never measured, so its time is not "
                f"in this total")
        if projection["fits_resident"]:
            add("    the whole model fits resident: this run is device-bandwidth bound, which is "
                "the best case this engine has.")
        else:
            add("    this is a projection from measured bandwidths, not a benchmark. Run "
                "tests/bench_streaming.py for the real number on this machine.")
    add("")
    add(rule)
    return "\n".join(lines)


def run(weights_path=None, device=None, reprofile=False, model=None, model_bytes=None,
        params=None, weight_bits=None, as_json=False, storage_budget_seconds=3.0, out=None):
    """Collect and print. Returns the collected data so tests do not have to parse text."""
    import sys

    stream = out if out is not None else sys.stdout
    collected = collect(weights_path=weights_path, device=device, reprofile=reprofile, model=model,
                        model_bytes=model_bytes, params=params, weight_bits=weight_bits,
                        storage_budget_seconds=storage_budget_seconds)
    if as_json:
        json.dump(to_dict(collected), stream, indent=2, default=str)
        stream.write("\n")
    else:
        stream.write(report(collected) + "\n")
    return collected
