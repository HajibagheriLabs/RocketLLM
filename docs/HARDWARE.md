# Hardware

RocketLLM has no reference machine. Nothing in the engine picks a number: every value that depends
on the hardware is measured or queried at runtime and flows out of `rocketllm/hw/profile.py`, and
every device-specific path is gated on a capability *query* rather than on a device name. A constant
that was right on the box it was tuned on is a bug on every other one.

This document is the contract that follows from that: which hardware is supported and how well, what
degrades when a capability is missing, how to read `rocketllm doctor`, and how to contribute a
result from your own machine.

## Support tiers

| Tier | Hardware | What you get |
| --- | --- | --- |
| 1 | CUDA, compute capability >= 8.0 | The primary target. bf16, fused 4-bit kernels where a kernel package is installed, pinned host staging, a dedicated copy stream, the full streaming pipeline. |
| 2 | CUDA CC 7.0–7.5; ROCm / AMD | Supported with documented fallbacks. Usually fp16 rather than bf16, and usually no fused 4-bit kernel, so packed weights are expanded into scratch. |
| 3 | Apple Silicon (MPS, and the separate MLX path); Intel XPU | Works, degraded. No pinned host memory and no async copy streams on MPS, so transfers are synchronous and do not overlap compute. Memory accounting is deliberately conservative on unified memory. |
| 4 | CPU | Correctness and CI only. Not a performance target: there is no separate device pool, so the device tier is empty and everything streams. |

A tier is a statement about capability, so it is derived from the capability. `CudaCaps.tier` reads
the compute capability and answers 1 or 2 from that — it never looks at the device's marketing name,
which tells you nothing about a card released next month.

## Capability gates and what happens without them

Each of these is answered by attempting the thing, not by assuming it from the backend. Each has a
defined fallback that still produces correct output, and each fallback is announced exactly once per
process — never once per layer, because a streaming run touches every module hundreds of times per
token and a per-layer notice would bury the console.

| Capability | How it is decided | Without it | What it costs |
| --- | --- | --- | --- |
| bf16 | A real bfloat16 matmul is attempted on the device | fp16, with a loud warning | **The one degradation whose symptom is wrong output rather than a slow run.** fp16's exponent range overflows to inf/NaN on very deep models, and the corruption surfaces as plausible-looking wrong tokens rather than as an error. If your output looks wrong, suspect this first. |
| fp16 | A real float16 matmul is attempted | fp32 | Correct, and twice the bytes moved per token. |
| fp8 | `torch._scaled_mm` is called on fp8 operands | fp8 checkpoints are read, then computed in the compute dtype | Storage is unaffected; the matmul is not fused. |
| Native fp4 | An fp4 tensor is allocated | 4-bit formats are dequantized before the matmul | This is the usual path. Native fp4 needs CC >= 10.0. |
| Fused 4-bit matmul | A kernel package imports **and** the device can run it | Packed weights are expanded into a small reusable scratch buffer immediately before the layer runs, and freed after | Extra device memory traffic per layer. Identical numbers. |
| Pinned host memory | A page-locked host buffer is actually allocated | Pageable staging buffers | The same bytes, transferred more slowly. Running *out* of pinned memory mid-run degrades the same way rather than failing. |
| Shard handles held open | `caps.commit_headroom()` — the OS's own commit accounting, then psutil, then plain host RAM | A small LRU pool of open shards instead of all of them | Only what stays *mapped* changes; every byte read is the same. On an OS that charges memory mappings against a commit limit, holding one handle per shard charges the whole checkpoint — a 67GB model on an 18GB limit died inside `safe_open` while merely reading headers. Linux under default overcommit charges nothing for a read-only mapping and keeps the fast path. |
| Async copy streams | A copy stream is created on the backend | The synchronous transfer path | Reads no longer overlap compute, so a streaming pass costs its full time instead of hiding behind the previous layer. |
| Triton | The `triton` package imports | The PyTorch dequant implementation | Slower dequantization, identical numbers. |

Two more failure modes are about memory rather than features:

- **The pin budget computes to zero.** The prefetch window's room is committed before anything is
  pinned, so on a small device the subtraction legitimately leaves nothing. That is pure streaming:
  every weight is read from storage on every token. It is correct, it is slow, and it is a supported
  configuration that the test suite exercises directly.
- **Not even one layer fits.** The window is floored at one layer, because a cache that cannot hold
  a single layer cannot run a forward pass at all. If even that will not fit, you get a clear error
  rather than a crash inside a matmul.

## What decides your speed

Time per token is approximately the sum, over memory tiers, of the bytes that tier has to serve
divided by that tier's measured bandwidth. The tiers, fastest to slowest, are VRAM, host RAM across
the PCIe link, NVMe, SATA SSD, and spinning disk.

Which tier dominates is entirely a question of how much of your model fits where, and that differs
per machine by orders of magnitude. A model that fits in VRAM is bandwidth-bound and fast. A model
ten times the size of VRAM is storage-bound, and the single most useful thing you can do about it is
move the weights onto a faster device.

The engine's whole job is to push bytes further up that hierarchy (residency, pinning, a host tier)
and to move fewer of them (4-bit packed weights, MoE experts instead of whole layers, more tokens
per streaming pass).

## Reading `rocketllm doctor`

```bash
rocketllm doctor --model /path/to/your/model
```

This is what to paste into a bug report. It prints six sections, and the interesting ones are rarely
the first.

**Hardware profile.** Every probed measurement, and every derived tuning knob printed next to the
formula that produced it and the inputs that went in. If a number looks wrong, the formula above it
usually shows why without anyone reading the source. `[OVERRIDDEN]` next to a knob means an
environment variable or a command-line flag replaced the measured value.

**Capability decisions.** One row per gate: the answer, what was attempted to decide it, and — only
where the answer was no — the fallback that was taken instead. Read the `NO` rows. A `NO` against
bf16 with an fp16 fallback is the single most common explanation for output that looks subtly wrong.

**Quantized checkpoint formats.** What each supported format would do on this machine:
`fused_packed` means weights stay packed through the matmul, `dequant_to_scratch` means they are
expanded first. Every row saying `dequant_to_scratch` on a machine that should have a fused kernel
means a kernel package is missing or does not import.

**Optional packages.** What is installed, what each one unlocks, and — for the missing ones — what
their absence costs and the command that installs them. Half of all reports are a missing kernel
package, and this section is the one that shows it.

**Weight storage.** The measured read bandwidth of the filesystem your weights are actually on,
the concurrency it saturates at, and whether the OS calls the device rotational. Lines beginning
`!!` are the loud ones. On a streaming engine a rotational or merely slow device dominates
everything else by orders of magnitude, and no amount of tuning will make up for it.

Two caveats the section states for itself. On platforms that cannot drop the page cache (anything
but Linux), some of those reads may have been served from RAM, so the number is optimistic. And if
no checkpoint shards were found at the path, the figure came from a temporary file written on the
same filesystem — it describes the right device but not the real read pattern.

**Projected cost per token.** Where a model of the given size would live on this machine, and what
one token would therefore cost, computed from the measured bandwidths through the model above. The
`share` column is the useful one: it tells you which tier you are actually bound by.

This is a projection, not a benchmark. Any tier whose bandwidth was never measured is reported as
`unavailable` and the total is labelled `INCOMPLETE` rather than being quietly completed with a
plausible number. For a real figure, run the benchmark harness.

Pass the model in whichever form you have:

```bash
rocketllm doctor --model /path/to/checkpoint    # measures the real on-disk size (best)
rocketllm doctor --model-bytes 40GB             # a size you already know
rocketllm doctor --model-size 70B --weight-bits 4
rocketllm doctor --json                         # the same content, machine-readable
```

## Overriding what was measured

Every tuning knob has a profile-derived default and a manual override, in that order. The overrides
exist to reproduce a problem or to bisect a measurement you suspect, not to configure a healthy run
— the derived value is the one you want, because it was measured on your machine and the override
was not.

Environment variables take the form `ROCKETLLM_<KNOB>`, for example:

```bash
ROCKETLLM_RESERVE_BYTES=2147483648 rocketllm serve --model ...
ROCKETLLM_IO_WORKERS=1 rocketllm serve --model ...
ROCKETLLM_HOST_CACHE_BYTES=0 rocketllm serve --model ...     # no host tier at all
```

`rocketllm profile` prints every knob that can be overridden along with what it currently came out
as. The same values are also exposed as flags on `rocketllm serve` and as keyword arguments to
`AutoModel.from_pretrained`.

One environment variable is not a RocketLLM knob but matters as much as any of them:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Set it before the first CUDA allocation. Under a streaming workload a layer's blocks are carved and
released hundreds of times per token, and this setting is the difference between reusing those
blocks and stranding them. `rocketllm doctor` warns when it is unset.

## Contributing a result from your machine

The community table in the README is meant to be filled in by machines the maintainer does not own,
which is most of them. A submission is worth more than a benchmark of the same model on a familiar
card, because the interesting question is not how fast RocketLLM is — it is which tier dominates on
which hardware, and that only different hardware can answer.

### Running the benchmark

```bash
python tests/bench_streaming.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --json
```

This writes `bench_results/<timestamp>_<model>_<profile fingerprint>.json`, which carries the whole
hardware profile alongside the numbers — the result and the machine it came from cannot be
separated, which is the point.

To compare against an earlier run on the same machine:

```bash
python tests/bench_streaming.py --model <same model> --compare-to bench_results/<previous>.json
```

To emulate a smaller card on a big one (CUDA only — it caps what the allocator will hand out):

```bash
python tests/bench_streaming.py --model <model> --max-vram-gb 4 --json
```

### What to include

The headline number is **bytes read per token, broken down by memory tier** — device, host, storage.
Tokens per second on its own says more about the machine than about the engine, and two machines
with the same tok/s can be in completely different regimes.

Open an issue or a pull request with:

- the JSON file the harness wrote, or its contents,
- the output of `rocketllm doctor --model <the model you benchmarked>`,
- the model and its quantization format,
- anything unusual about the machine: an external drive, a laptop that thermally throttles, a card
  shared with a display.

A result that is *slow* is as useful as a fast one, and a result where something degraded
unexpectedly is the most useful of all. If `rocketllm doctor` warned about something and you think
the warning is wrong, that is a bug report and it is welcome.

### Reporting a bug

Include the full `rocketllm doctor` output. It is designed to be pasted whole, and without it a
report of "it is slow" or "the output is wrong" cannot be distinguished from a hardware limit — the
two look identical from the outside and only one of them is actionable.

## Testing hardware you do not have

The suite runs on a plain CPU runner with no accelerator, and that property is deliberate: a test
that only passes on the maintainer's machine tells a contributor nothing about theirs.

```bash
pytest tests/                        # everything, no accelerator required
pytest tests/test_portability.py     # the emulated hardware matrix
pytest tests/test_cpu_generation.py  # the correctness gate, on the CPU backend
```

`tests/test_portability.py` describes machines rather than requiring them. A `HardwareProfile` whose
measurements are supplied instead of probed *is* a different machine as far as the engine can tell,
since the derived knobs are what the cache sizes itself from. The matrix sweeps devices that hold
the whole model, about half of it, exactly one layer, and none of it, along with devices missing
pinned memory, copy streams, fused kernels and bf16. Where a real CUDA device is present, the same
cases additionally cap the process so the allocator genuinely refuses.

Adding a case is a matter of describing the device — see `EmulatedDevice` — and the assertion is
always the same: the generated token ids must be identical to a full load of the same weights.
Where a device is short of something, it is allowed to be slower. It is never allowed to be wrong.

If you have hardware in a tier that is under-tested — anything ROCm, Intel XPU, Apple Silicon, or a
pre-Ampere NVIDIA card — running the real suite on it and reporting what happened is the single most
valuable contribution available:

```bash
ROCKETLLM_TEST_DEVICE=cuda:0 pytest tests/test_portability.py tests/test_cpu_generation.py -q
python tests/test_streaming_gpu.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --compare
```

The second command is the correctness gate proper: it runs the model both ways and must print
`MATCH`. A fast engine that produces the wrong tokens is worth nothing.
