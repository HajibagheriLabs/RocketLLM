"""Command line entry point.

``rocketllm profile`` exists so that a bug report can carry the machine it happened on. It prints
every probed measurement and every derived knob next to the formula that produced it, which is
usually enough to see why a number came out the way it did without anyone reading the source.

``rocketllm doctor`` is what to ask a bug reporter for. It is the profile plus the things the
profile alone does not answer: which capability gates said no and what was taken instead, which
optional packages are missing and what each absence costs, how fast the weights' filesystem
actually measured, and what a model of a given size would therefore cost per token on this machine.

``rocketllm serve`` runs the OpenAI-compatible server. Every tuning flag it takes defaults to None,
meaning "use what the hardware profile measured on this machine" -- the flags exist to reproduce a
problem or bisect a suspect measurement, not to configure a healthy run. There is no default that is
a number, because there is no reference machine for a number to have been right on.
"""
import argparse
import json
import sys


# ---- profile ------------------------------------------------------------------------------------

def _add_profile_parser(subparsers):
    parser = subparsers.add_parser(
        "profile",
        help="probe this machine and print the hardware profile and every derived tuning knob")
    parser.add_argument("--weights-path", default=None,
                        help="probe storage on the filesystem holding these weights; without it "
                             "storage bandwidth cannot be measured and is reported as unavailable")
    parser.add_argument("--device", default=None,
                        help="override the auto-selected backend, e.g. cpu or cuda:1")
    parser.add_argument("--reprofile", action="store_true",
                        help="ignore any cached profile and measure again")
    parser.add_argument("--json", action="store_true",
                        help="emit the profile as JSON instead of the readable report")
    parser.add_argument("--storage-budget-seconds", type=float, default=3.0,
                        help="time budget for the storage sweep (default: 3.0)")
    parser.set_defaults(handler=_run_profile)
    return parser


def _run_profile(args):
    from .hw import HardwareProfile

    profile = HardwareProfile.load_or_probe(
        weights_path=args.weights_path,
        device=args.device,
        reprofile=args.reprofile,
        storage_budget_seconds=args.storage_budget_seconds)

    if args.json:
        json.dump(profile.to_dict(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(profile.describe())
    return 0


# ---- shared argument parsing --------------------------------------------------------------------

_SIZE_SUFFIXES = {"k": 1024, "kb": 1024, "m": 1024 ** 2, "mb": 1024 ** 2,
                  "g": 1024 ** 3, "gb": 1024 ** 3, "t": 1024 ** 4, "tb": 1024 ** 4}

_COUNT_SUFFIXES = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def size_bytes(text):
    """Parse a byte count, with or without a unit: 2GB, 512mb, 1073741824."""
    raw = str(text).strip().lower().replace("i", "")  # GiB and GB mean the same thing here
    for suffix, scale in sorted(_SIZE_SUFFIXES.items(), key=lambda kv: -len(kv[0])):
        if raw.endswith(suffix):
            raw = raw[:-len(suffix)].strip()
            break
    else:
        scale = 1
    try:
        return int(float(raw) * scale)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a byte size; write it as 2GB, 512MB or a plain number of bytes")


def param_count(text):
    """Parse a parameter count the way people write one: 70B, 8b, 1.5M, 405000000."""
    raw = str(text).strip().lower()
    scale = _COUNT_SUFFIXES.get(raw[-1:], 1)
    if scale != 1:
        raw = raw[:-1].strip()
    try:
        return int(float(raw) * scale)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a parameter count; write it as 70B, 8b or a plain number")


def torch_dtype(name):
    if name in (None, "auto"):
        return None
    import torch

    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise argparse.ArgumentTypeError(f"{name!r} is not a torch dtype")
    return dtype


# ---- doctor -------------------------------------------------------------------------------------

def _add_doctor_parser(subparsers):
    parser = subparsers.add_parser(
        "doctor",
        help="print everything a bug report needs: profile, capability decisions, optional "
             "packages, storage bandwidth and a projected per-token cost",
        description="Diagnose this machine. Run it and paste the output into an issue -- RocketLLM "
                    "has no reference machine, so what it did on yours is the only way to tell a "
                    "bug from a hardware limit.")
    parser.add_argument("--model", default=None,
                        help="a checkpoint to size the projection against, and the filesystem to "
                             "measure storage on. The most useful thing to pass: the model's real "
                             "on-disk size is what the engine actually has to move")
    parser.add_argument("--weights-path", default=None,
                        help="measure storage here instead of at --model, for when the weights are "
                             "somewhere other than the model you are asking about")
    parser.add_argument("--model-bytes", type=size_bytes, default=None,
                        help="project for a model of this many bytes, e.g. 40GB, when the "
                             "checkpoint is not on this machine")
    parser.add_argument("--model-size", type=param_count, default=None, dest="params",
                        help="project for a model of this many parameters, e.g. 70B. Combined with "
                             "--weight-bits; less accurate than --model, which measures the file")
    parser.add_argument("--weight-bits", type=int, default=None,
                        help="bits per weight in the checkpoint --model-size describes "
                             "(default: 16, i.e. an unquantized checkpoint)")
    parser.add_argument("--device", default=None,
                        help="diagnose this backend instead of the auto-selected one, e.g. cpu")
    parser.add_argument("--reprofile", action="store_true",
                        help="ignore any cached profile and measure this machine again")
    parser.add_argument("--json", action="store_true",
                        help="emit the whole diagnosis as JSON instead of the readable report")
    parser.add_argument("--storage-budget-seconds", type=float, default=3.0,
                        help="time budget for the storage sweep (default: 3.0)")
    parser.set_defaults(handler=_run_doctor)
    return parser


def _run_doctor(args):
    from .hw import doctor

    doctor.run(weights_path=args.weights_path, device=args.device, reprofile=args.reprofile,
               model=args.model, model_bytes=args.model_bytes, params=args.params,
               weight_bits=args.weight_bits, as_json=args.json,
               storage_budget_seconds=args.storage_budget_seconds)
    return 0


# ---- serve --------------------------------------------------------------------------------------

def _add_serve_parser(subparsers):
    parser = subparsers.add_parser(
        "serve", help="run the OpenAI-compatible HTTP server against a model",
        description="Serve a model over an OpenAI-compatible HTTP API. One request is generated at "
                    "a time; concurrent requests queue.")
    parser.add_argument("--model", required=True,
                        help="local path or Hugging Face repo id of the model to serve")
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind (default: 127.0.0.1, i.e. this machine only; pass "
                             "0.0.0.0 to accept connections from the network)")
    parser.add_argument("--port", type=int, default=8000, help="port to bind (default: 8000)")
    parser.add_argument("--served-model-name", default=None,
                        help="the id reported by /v1/models and echoed in responses "
                             "(default: the model directory's name)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="server-wide ceiling on one reply. Default: none, so a request may use "
                             "whatever is left of the model's context")
    parser.add_argument("--log-level", default="info",
                        choices=["critical", "error", "warning", "info", "debug", "trace"],
                        help="uvicorn log level (default: info)")

    model = parser.add_argument_group(
        "model", "how the model itself is loaded")
    model.add_argument("--device", default=None,
                       help="backend to run on, e.g. cuda:0, mps, cpu. Default: auto-selected")
    model.add_argument("--dtype", type=torch_dtype, default=None,
                       help="compute dtype (float16, bfloat16, float32). Default: the checkpoint's "
                            "own, degraded to what this device actually supports")
    model.add_argument("--max-seq-len", type=int, default=512,
                       help="sequence length the streaming engine is set up for (default: 512)")
    model.add_argument("--hf-token", default=None, help="Hugging Face token for gated repos")
    model.add_argument("--layer-shards-path", default=None,
                       help="where to write the per-layer shards (default: beside the model cache)")
    model.add_argument("--delete-original", action="store_true",
                       help="delete the downloaded checkpoint once it has been split")
    model.add_argument("--no-prefetching", dest="prefetching", action="store_false",
                       help="disable overlapping the next layers' reads with the current compute")

    tuning = parser.add_argument_group(
        "tuning overrides",
        "Every one of these defaults to the value HardwareProfile measured on THIS machine. They "
        "are debugging levers: a number that was right on the box it was chosen on is wrong on the "
        "next one. Run `rocketllm profile` to see what the defaults came out as and why.")
    tuning.add_argument("--vram-reserve", type=size_bytes, default=None,
                        help="device memory held back for activations, workspace and "
                             "fragmentation. Default: profile reserve_bytes")
    tuning.add_argument("--host-cache-gb", type=float, default=None,
                        help="gigabytes of host RAM the cache may use as its middle tier. Zero is "
                             "valid and means evictions drop straight to storage. Default: profile "
                             "host_cache_bytes")
    tuning.add_argument("--io-workers", type=int, default=None,
                        help="concurrent storage readers. Default: the concurrency measured to "
                             "saturate this machine's storage")
    tuning.add_argument("--window-max", type=int, default=None,
                        help="hard cap on decoder layers held in the prefetch window. Default: the "
                             "window budget divided by the largest layer")
    tuning.add_argument("--pin-policy", choices=["auto", "off"], default="auto",
                        help="'auto' fills the pin budget by bytes-saved-per-resident-byte; 'off' "
                             "pins nothing and streams everything (default: auto)")
    tuning.add_argument("--expert-residency", choices=["auto", "off"], default="auto",
                        help="keep popular MoE experts resident across tokens (default: auto)")
    tuning.add_argument("--kv-cache", default="auto",
                        help="how to hold the KV cache: auto, fp16, int4, hqq, quanto. 'auto' "
                             "quantizes only when the weights do not fit resident (default: auto)")
    tuning.add_argument("--draft-model", default=None,
                        help="a small model sharing this one's tokenizer, kept resident to propose "
                             "tokens for speculative decoding. Nothing happens without one")
    tuning.add_argument("--speculative", choices=["auto", "on", "off"], default="auto",
                        help="'auto' follows the profile's measured recommendation (default: auto)")
    tuning.add_argument("--prefix-cache", choices=["auto", "on", "off"], default="auto",
                        help="reuse the KV cache of a prefix already seen instead of re-prefilling "
                             "the conversation every turn. 'auto' follows the measured "
                             "recommendation: a prefill is one streaming pass, so reuse pays where "
                             "the weights are resident and costs more than it saves where they are "
                             "not (default: auto)")
    tuning.add_argument("--prefix-cache-gb", type=float, default=None,
                        help="gigabytes of host RAM the prefix cache may hold. Default: profile "
                             "prefix_cache_bytes. Zero disables it")
    tuning.add_argument("--tool-parser", default=None,
                        help="force the tool-call syntax to read out of replies. Default: detected "
                             "from what the model's own chat template emits, which is right for "
                             "any checkpoint whose template renders tool calls")

    parser.set_defaults(handler=_run_serve, prefetching=True)
    return parser


def _run_serve(args):
    # Imported here, not at module scope: `rocketllm profile` must keep working on a machine that
    # has not installed the server extra, and importing it would be the only thing that broke.
    try:
        from .server import GenerationEngine, create_app
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1
    import uvicorn

    from .auto_model import AutoModel
    from .hw.caps import resolve_device

    # Not defaulted to "cuda:0". The device is queried, like every other hardware fact here, so
    # serving on a machine without an NVIDIA card is a slower run rather than a crash on startup.
    device = str(resolve_device(args.device))

    model = AutoModel.from_pretrained(
        args.model,
        device=device,
        dtype=args.dtype,
        max_seq_len=args.max_seq_len,
        layer_shards_saving_path=args.layer_shards_path,
        hf_token=args.hf_token,
        prefetching=args.prefetching,
        delete_original=args.delete_original,
        vram_reserve=args.vram_reserve,
        host_cache_gb=args.host_cache_gb,
        io_workers=args.io_workers,
        window_max=args.window_max,
        pin_policy=args.pin_policy,
        expert_residency=args.expert_residency,
        kv_cache=args.kv_cache,
        draft_model=args.draft_model,
        speculative=args.speculative)

    # The id a client sees defaults to what the user asked for, not to where the weights landed:
    # a downloaded checkpoint lives under a commit hash, and answering requests as that helps no one.
    engine = GenerationEngine(
        model, model_id=args.served_model_name or args.model, max_tokens=args.max_tokens,
        tool_parser=args.tool_parser, prefix_cache=args.prefix_cache,
        prefix_cache_bytes=(None if args.prefix_cache_gb is None
                            else int(args.prefix_cache_gb * 1024 ** 3)))
    app = create_app(engine)
    print(f"serving {engine.model_id} on http://{args.host}:{args.port}  "
          f"(context {engine.context_length} tokens, one request at a time, "
          f"tool calls parsed as {engine.tool_parser.family})")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    finally:
        model.close()
    return 0


# ---- entry point --------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(prog="rocketllm", description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command")
    _add_profile_parser(subparsers)
    _add_doctor_parser(subparsers)
    _add_serve_parser(subparsers)

    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 1
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
