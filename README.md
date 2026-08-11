# RocketLLM

**Run language models far larger than your VRAM.**

RocketLLM is an inference engine for models that do not fit on your GPU. Instead of requiring the
whole checkpoint to be resident, it keeps the model on storage and streams it through the device
layer by layer — and, for mixture-of-experts models, expert by expert — holding as much in fast
memory as the machine actually has room for. The model is built on PyTorch's `meta` device, so it
costs no memory until a layer is about to run, and `transformers` still owns the forward pass, which
means new architectures generally work without changes to RocketLLM.

There is no reference machine. RocketLLM measures the hardware it finds itself on — VRAM, host RAM,
transfer bandwidth, storage bandwidth, device capability — and derives its own tuning from that,
because a constant that is right on one box is wrong on every other one.

Author: Hadi Hajibagheri.

## The performance model

Time per token is approximately the sum, over every memory tier, of the bytes served from that tier
divided by that tier's measured bandwidth — where the tiers, fastest to slowest, are VRAM, host RAM
over PCIe, NVMe, SATA SSD, and spinning disk.

Which tier dominates depends entirely on how much of your model fits where, so the only way to go
faster is to move fewer bytes (4-bit packed weights, only the routed experts), to serve those bytes
from a higher tier (resident layers, pinned host memory), or to amortize one streaming pass over more
tokens.

## Device support

RocketLLM gates every device-specific path on a capability *query*, never on a device name, and every
missing capability has a defined fallback that still produces correct output.

| Tier | Hardware | Status |
| --- | --- | --- |
| 1 | CUDA, compute capability >= 8.0 | Primary target. bf16, fused 4-bit kernels, full streaming pipeline. |
| 2 | CUDA CC 7.0–7.5; ROCm / AMD | Supported, with documented fallbacks (fp16 in place of bf16, dequantize in place of fused kernels). |
| 3 | Apple Silicon (MPS and the MLX path); Intel XPU | Works, degraded. No pinned host memory, no async copy streams. |
| 4 | CPU | Correctness and CI only. Not a performance target. |

Where a capability is absent, RocketLLM logs the fallback once — not once per layer — and keeps
running. Notably, if bf16 is unavailable it falls back to fp16 with an explicit warning: fp16's range
overflows on very deep models and can silently corrupt output.

## Install

```bash
pip install rocketllm
```

From a checkout:

```bash
pip install -e .
```

Optional extras, installed only if you need them:

```bash
pip install "rocketllm[quant]"
```

`quant` pulls in the readers for bitsandbytes-prequantized and compressed-tensors checkpoints; `mlx`
pulls in the Apple Silicon backend; `server` pulls in FastAPI, uvicorn and pydantic for
`rocketllm serve`. None is required for a base install.

## Quickstart

```python
from rocketllm import AutoModel

model = AutoModel.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

inputs = model.tokenizer(["The capital of France is"],
                         return_tensors="pt",
                         return_attention_mask=False)

out = model.generate(inputs["input_ids"].cuda(),
                     max_new_tokens=20,
                     use_cache=True,
                     return_dict_in_generate=True)

print(model.tokenizer.decode(out.sequences[0]))
```

`AutoModel` reads the checkpoint's architecture and picks the right class. Most models use the
generic `RocketModel`; a handful of custom-architecture families (ChatGLM, QWen, Baichuan, InternLM,
Kimi K3) have dedicated subclasses that only override module naming. On first use the checkpoint is
split into per-layer shards next to the model cache; subsequent runs reuse them.

More examples, including Llama 3.1 405B and the Apple Silicon path, are in [`examples/`](examples/).

## OpenAI-compatible server

```bash
pip install "rocketllm[server]"
rocketllm serve --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
       "messages": [{"role": "user", "content": "Name three primary colours."}],
       "max_tokens": 40}'
```

Add `"stream": true` for server-sent events, terminated by `data: [DONE]`. Any OpenAI client works
by pointing its base URL at `http://127.0.0.1:8000/v1`.

| Endpoint | What it does |
| --- | --- |
| `POST /v1/chat/completions` | Chat, streaming or not. Renders the model's own chat template. |
| `POST /v1/completions` | Raw-prompt completion, streaming or not. |
| `GET /v1/models` | The loaded model, plus its context length, device, dtype and KV cache mode. |
| `GET /health` | Hardware profile, cache statistics, queue depth. Paste this into bug reports. |

Sampling: `temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `seed`, `repetition_penalty`.
`presence_penalty` and `frequency_penalty` are accepted for client compatibility but not applied,
because transformers has no equivalent with the same meaning — the server says so once rather than
pretending they took effect.

**One request at a time.** There is one model instance streaming one set of weights, so concurrent
requests queue rather than batch; each is told its position in the `x-rocketllm-queue-position`
response header. Continuous batching is deliberately out of scope: two sequences sharing this engine
would evict each other's layers from the weight cache and both run slower than either alone. A client
that hangs up has its generation cancelled, so an abandoned request does not hold the worker.

`rocketllm serve` takes the same tuning overrides as the Python API — `--vram-reserve`,
`--host-cache-gb`, `--io-workers`, `--window-max`, `--pin-policy`, `--expert-residency`,
`--kv-cache`, `--draft-model`, `--speculative` — and every one of them defaults to what
`rocketllm profile` measured on your machine. Run `rocketllm serve --help` for the full list.

## Supported quantization formats

RocketLLM *imports* pre-quantized checkpoints. It does not quantize models itself.

| Format | Notes |
| --- | --- |
| AWQ | 4-bit weights. |
| GPTQ | 4-bit weights. |
| compressed-tensors W4A16 | Requires the `compressed-tensors` reader. |
| MXFP4 | Used by large MoE checkpoints. |
| bitsandbytes (pre-quantized) | Loaded as stored; RocketLLM does not quantize on the fly. |

Where fused kernels exist for the running device, weights stay packed and are computed on directly.
Where they do not, they are dequantized into a small reusable scratch buffer immediately before the
layer runs. That choice is made by capability query, not by format name.

KV cache quantization is independent of the weight format: int4, per-channel for K and per-token for
V, with an fp16 residual window over the most recent tokens.

## Benchmarks

The headline number for RocketLLM is **bytes read per token, broken down by memory tier** — tokens
per second on its own says more about the machine than about the engine.

Results differ by orders of magnitude across hardware, so this table is meant to be filled in by
community submissions rather than by one machine. The benchmark harness that produces a row lands
with the streaming rework; until then the table stays empty rather than carrying numbers from a
single box.

| GPU | VRAM | Host RAM | Weight storage | Model | Quant | Bytes/token (VRAM / host / storage) | tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| _awaiting community submissions_ | | | | | | | |

## Roadmap

- Runtime hardware profiling that derives every tuning knob, with manual overrides.
- Tiered weight cache: statically pinned layer subset plus a FIFO prefetch window for dense layers,
  LFU-with-aging residency for MoE experts.
- Coalesced storage reads into pooled host buffers, with a dedicated copy stream where one exists.
- int4 KV cache.
- Speculative decoding.
- Tool calling and prefix-KV reuse for the server.
- A portability matrix covering the four device tiers.

## Credits

RocketLLM began as a fork of [AirLLM](https://github.com/lyogavin/airllm) by Gavin Li and the AirLLM
contributors, which originated the layer-streaming approach this engine is built on. RocketLLM has
since been substantially rewritten, and is an independent project — not affiliated with or endorsed
by AirLLM or its authors.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Attribution is binding: [NOTICE](NOTICE) must be retained and reproduced in any distribution or
derivative work, as required by Section 4(d) of the license.
