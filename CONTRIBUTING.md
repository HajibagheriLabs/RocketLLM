# Contributing

Thanks for looking. RocketLLM has no reference machine and its contributors run wildly different
hardware, which shapes almost everything below.

## The two rules

**1. Never hardcode a hardware value.** Not GPU model or vendor, VRAM size, host RAM, storage type or
bandwidth, PCIe generation, compute capability, dtype support, or kernel availability. Every one of
those is measured or queried at runtime and comes from `rocketllm/hw/profile.py`. If you need a
number that is not in the profile yet, add it to the profile rather than picking a constant. A
constant that was right on the box it was chosen on is a bug on every other one.

Anything device-specific must be gated on a capability *query*, never on a device name, and must have
a defined fallback that still produces correct output. A missing hardware feature is a slower path,
never an error.

**2. Never claim a speedup you have not measured.** The primary metric is *bytes read per token,
broken down by memory tier*. Report the delta, and say which tier the bytes came from. If a change
did not improve the numbers, say so plainly.

## Setting up

```bash
python -m pip install -e ".[dev]"
```

CPU-only PyTorch is enough for everything except the GPU correctness gate:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Before you open a pull request

```bash
ruff check .
```
```bash
pytest tests/ -q
```

Both must be clean. The suite runs with no accelerator and no network — keep that property, because
it is what lets a contributor verify their own change.

If you touched a device-specific path, also run the emulated hardware matrix explicitly:

```bash
pytest tests/test_portability.py -q
```

If you have an accelerator, run the real correctness gate. It must print `MATCH`:

```bash
python tests/test_streaming_gpu.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --compare
```

If your change was meant to make something faster, measure it:

```bash
python tests/bench_streaming.py --model <model> --json --compare-to bench_results/<previous>.json
```

Paste the delta table into the pull request along with the hardware profile it was measured on. A
benchmark without its machine cannot be compared to anything.

## What a change has to prove

- **Correctness first.** Streaming changes *where* a weight is when the matmul reads it, never what
  the model computes. Streamed output is compared to a full load and must be identical, not close.
- **It works on hardware you do not have.** If a change touches a capability gate, a memory budget or
  a cache policy, add a case to `tests/test_portability.py`. Describing a machine costs nothing;
  waiting for someone with that machine to report a bug costs a release.
- **Degradation is tested, not assumed.** Any new optional dependency or device feature needs the
  absent path exercised.

A red suite is never merged, and never worked around by relaxing an assertion or marking a test
skipped. If a test genuinely cannot apply — the architecture it covers does not exist in the oldest
supported `transformers`, say — skip it on a condition that names the reason.

## Things that look like improvements and are not

Some designs here are counterintuitive and load-bearing. Before simplifying one, read
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), which has a section listing them and why. The short
version: dense layers are deliberately not LRU (a cyclic scan defeats recency completely), the cache
stores packed bytes, the hysteresis band is a share rather than a byte count, and there is no
cross-layer MoE router lookahead to design around.

## Adding a backend

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has step-by-step instructions for adding a quantization
backend and for adding a device backend, including which methods have working fallbacks you do not
need to override.

## Reporting a bug

Open an issue and include the full output of:

```bash
rocketllm doctor --model /path/to/your/model
```

This is not a formality. "It is slow" and "the output is wrong" are indistinguishable from a hardware
limit without it, and only one of those is actionable. The doctor prints the machine, every
capability decision and the fallback taken for each, which optional packages are missing, and the
measured bandwidth of the filesystem your weights are on — which is frequently the entire answer.

## Contributing a benchmark result

Results from hardware the author does not own are the most valuable contribution available. A slow
result is as useful as a fast one, and one where something degraded unexpectedly is the most useful
of all. See
[docs/HARDWARE.md](docs/HARDWARE.md#contributing-a-result-from-your-machine).

## Style

Match the surrounding file. Comments explain *why* a non-obvious decision was made, not what the line
does — most of the hard-won knowledge in this codebase lives in those comments, and a change that
invalidates one should update it. Lint is `ruff check .` at 120 columns over `E`/`F`/`W`; formatting
opinions are deliberately not enforced.

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0, the same
terms as the project. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
