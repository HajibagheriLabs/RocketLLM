"""
Manual GPU test harness for RocketLLM layer-streaming inference.

Runs a model through RocketLLM (one layer on the GPU at a time) and reports peak VRAM.
Optionally caps the visible VRAM to emulate a small card, and/or compares the output
against a normal full-load run of the same model for correctness.

Examples
--------
# Tiny model, verify output matches a normal full-load run, report peak VRAM:
python test_streaming_gpu.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --compare

# Emulate a 4GB card running a 7B model:
python test_streaming_gpu.py --model Qwen/Qwen2.5-7B-Instruct --max-vram-gb 4

# A real 70B on whatever card you have, capped to 8GB:
python test_streaming_gpu.py --model meta-llama/Llama-3.3-70B-Instruct --max-vram-gb 8 \
    --prompt "The capital of France is" --max-new-tokens 8
"""
import argparse
import sys
import time
from pathlib import Path

import torch

# Run as a plain script from a checkout, with or without an editable install: the package sits at
# the repo root, which is not on the path when this file is executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def cap_vram(max_vram_gb):
    """Limit how much of the GPU this process may allocate, to emulate a smaller card.

    Capping is a CUDA allocator feature, so it is gated on the capability rather than assumed:
    on every other backend the request is reported as unavailable and the run continues at the
    device's real size.
    """
    if max_vram_gb is None:
        return
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        print(f"--max-vram-gb {max_vram_gb} ignored: capping needs the CUDA allocator, "
              f"which is unavailable on this backend")
        return
    total = torch.cuda.get_device_properties(0).total_memory
    frac = (max_vram_gb * (1024 ** 3)) / total
    if frac >= 1.0:
        print(f"requested cap {max_vram_gb}GB >= device memory; not capping")
        return
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f"capped process VRAM to ~{max_vram_gb}GB ({frac*100:.1f}% of device)")


def run_rocketllm(args):
    from rocketllm import AutoModel

    model = AutoModel.from_pretrained(args.model, compression=args.compression,
                                      delete_original=args.delete_original)
    ids = model.tokenizer([args.prompt], return_tensors="pt",
                          return_attention_mask=False)["input_ids"].cuda()

    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                         return_dict_in_generate=True)
    elapsed = time.time() - t

    text = model.tokenizer.decode(out.sequences[0])
    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"\n=== RocketLLM ===")
    print(f"output : {text!r}")
    print(f"peak VRAM: {peak:.1f} MB")
    print(f"time   : {elapsed:.1f}s ({args.max_new_tokens} new tokens)")
    return out.sequences[0].tolist()


def run_reference(args):
    """Full-load run for correctness comparison (only feasible for small models)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    # transformers renamed this argument: `torch_dtype` up to 4.5x, `dtype` from 4.56 on, and the
    # old name is gone in 5.x. We support both ends of that range, so try the current spelling and
    # fall back rather than pinning the gate to one transformers generation.
    try:
        ref = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    except TypeError:
        ref = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    ref = ref.cuda().eval()
    ids = tok([args.prompt], return_tensors="pt", return_attention_mask=False)["input_ids"].cuda()
    out = ref.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    text = tok.decode(out[0])
    print(f"\n=== Reference (full load) ===")
    print(f"output : {text!r}")
    del ref
    torch.cuda.empty_cache()
    return out[0].tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--max-vram-gb", type=float, default=None)
    p.add_argument("--compression", default=None, choices=[None, "4bit", "8bit"])
    p.add_argument("--delete-original", action="store_true",
                   help="delete the original checkpoint shards while splitting (saves disk for big models)")
    p.add_argument("--compare", action="store_true",
                   help="also run a full-load reference and assert outputs match")
    args = p.parse_args()

    ref_seq = None
    if args.compare:
        # run reference first (before capping) so the full model fits
        ref_seq = run_reference(args)

    cap_vram(args.max_vram_gb)
    air_seq = run_rocketllm(args)

    if ref_seq is not None:
        match = ref_seq == air_seq
        print(f"\n=== correctness: {'MATCH' if match else 'MISMATCH'} ===")
        if not match:
            print(f"ref : {ref_seq}")
            print(f"air : {air_seq}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
