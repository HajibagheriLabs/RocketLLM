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

To see what your own machine decided, and what it will cost you:

```bash
rocketllm doctor --model /path/to/your/model
```

That prints the hardware profile, every capability decision and the fallback taken for each, which
optional packages are missing and what their absence costs, the measured read bandwidth of the
filesystem your weights are on, and a projected per-token cost. It is also what to paste into a bug
report. [docs/HARDWARE.md](docs/HARDWARE.md) explains the tiers, the degradations, how to read that
output, and how to contribute a benchmark result from your own hardware.

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

out = model.generate(inputs["input_ids"].to(model.device),
                     max_new_tokens=20,
                     use_cache=True,
                     return_dict_in_generate=True)

print(model.tokenizer.decode(out.sequences[0]))
```

`AutoModel` reads the checkpoint's architecture and picks the right class. Most models use the
generic `RocketModel`; a handful of custom-architecture families (ChatGLM, QWen, Baichuan, InternLM,
Kimi K3) have dedicated subclasses that only override module naming. On first use the checkpoint is
split into per-layer shards next to the model cache; subsequent runs reuse them.

`model.device` is the backend RocketLLM resolved for itself — use it rather than `.cuda()`, so the
same script runs on whatever the machine turns out to have.

More examples, including Llama 3.1 405B and the Apple Silicon path, are in [`examples/`](examples/).

## Configuration

Every option below defaults to `None` or `"auto"`, and every one of those defaults means **the value
`HardwareProfile` measured on this machine**. That is the setting you want. These exist to reproduce
a problem or bisect a measurement you suspect, not to tune a healthy run — a number that is right on
the box it was chosen on is wrong on the next one.

Run `rocketllm profile` to see what each one currently derives to, and the formula and inputs that
produced it. Anything you override is printed back marked `[OVERRIDDEN]`.

### `AutoModel.from_pretrained(...)` / `RocketModel(...)`

Model and checkpoint handling:

| Option | Default | What it does |
| --- | --- | --- |
| `model_local_path_or_repo_id` | *required* | Local checkpoint path or Hugging Face repo id. |
| `device` | **profile** — the fastest backend actually present | Backend to run on, e.g. `"cuda:0"`, `"mps"`, `"cpu"`. Queried, never assumed, so a machine with no accelerator runs slower rather than failing. |
| `dtype` | **profile** — the checkpoint's own dtype, degraded to what the device supports | Compute dtype. A checkpoint asking for bf16 on a device without bf16 becomes fp16 with a loud warning rather than being honoured as written. |
| `max_seq_len` | `512` | Sequence length the streaming engine is set up for. A model property, not a hardware one. |
| `layer_shards_saving_path` | `None` — beside the model cache | Where the per-layer shards are written. |
| `hf_token` | `None` | Hugging Face token, for gated repos. |
| `delete_original` | `False` | Delete the downloaded checkpoint once it has been split. Saves disk on very large models. |
| `prefetching` | `True` | Overlap the next layers' reads with the current layer's compute. |
| `profiling_mode` | `False` | Record per-phase timings. |
| `compression` | *removed* | Raises. RocketLLM imports pre-quantized checkpoints and does not quantize models itself; the error names the formats it reads. |

Tuning overrides — every default is a measurement:

| Option | Default | What it does |
| --- | --- | --- |
| `vram_reserve` | **profile** `reserve_bytes` | Device memory held back for activations, workspace and fragmentation. Derived from the allocator's *measured* fragmentation ratio and workspace high-water mark, not from a round number. |
| `host_cache_gb` | **profile** `host_cache_bytes` | Host RAM the cache may hold as its middle tier. A share of *available* RAM after OS headroom. Zero is valid and means evictions drop straight to storage. |
| `io_workers` | **profile** `io_workers` | Concurrent storage readers. The concurrency that was measured to saturate *this* machine's storage — one reader is latency-bound below a fast drive's rated bandwidth, too many thrash a slow one. |
| `window_max` | **profile** `window_budget_bytes` ÷ largest layer | Hard cap on decoder layers held in the prefetch window. Floored at one, because a cache that cannot hold a single layer cannot run a forward pass. |
| `pin_policy` | `"auto"` | `auto` fills the pin budget by bytes-saved-per-resident-byte; `off` pins nothing and streams everything. `off` is what a device with no spare memory gets anyway. |
| `expert_residency` | `"auto"` | `auto` counts how often each MoE expert is routed to and keeps the popular ones resident; `off` disables it. Exists so the policy can be measured against its own absence. |
| `kv_cache` | `"auto"` | `auto` keeps the context in the compute dtype when the weights fit resident with headroom, and quantizes to int4 when they do not — only in the second case is a byte of context a byte of weights re-read next token. `fp16` and `int4` force it; `hqq` and `quanto` delegate to transformers' own quantized caches. Independent of the weight format, always. |
| `draft_model` | `None` | A small model sharing this one's tokenizer, kept resident to propose tokens for speculative decoding. Vocabulary and tokenizer are checked at load and a mismatch is refused, because it produces plausible wrong output rather than an error. Nothing happens without one. |
| `speculative` | `"auto"` | `auto` follows the profile's measured recommendation: speculation pays when a streaming pass costs far more than device memory does, and costs residency when it does not. `on` and `off` force it. `auto` never enables it against the measurement, and never on a machine whose bandwidths could not be measured at all. |

### `rocketllm serve`

Takes every tuning override above as a flag, with the same profile-derived defaults, plus:

| Flag | Default | What it does |
| --- | --- | --- |
| `--model` | *required* | Local path or repo id to serve. |
| `--host` | `127.0.0.1` | Interface to bind. This machine only; pass `0.0.0.0` to accept connections from the network. |
| `--port` | `8000` | Port to bind. |
| `--served-model-name` | the model directory's name | The id reported by `/v1/models` and echoed in responses. A downloaded checkpoint lives under a commit hash, and answering as that helps no one. |
| `--max-tokens` | `None` | Server-wide ceiling on one reply. Default: a request may use whatever is left of the context. |
| `--max-seq-len` | `512` | Sequence length the engine is set up for. |
| `--log-level` | `info` | uvicorn log level. |
| `--device` `--dtype` | **profile** | As above. |
| `--vram-reserve` `--host-cache-gb` `--io-workers` `--window-max` | **profile** | As above. `--vram-reserve` accepts `2GB`, `512MB` or a plain byte count. |
| `--pin-policy` `--expert-residency` `--kv-cache` `--draft-model` `--speculative` | `auto` | As above. |
| `--prefix-cache` | `"auto"` | Reuse the KV cache of a prefix already seen instead of re-prefilling every turn. `auto` follows the measured recommendation — see below, it is off unless the weights fit resident. |
| `--prefix-cache-gb` | **profile** `prefix_cache_bytes` | Host RAM the prefix cache may hold. Zero disables it. |
| `--tool-parser` | detected from the chat template | Force the tool-call syntax to read out of replies. |
| `--no-prefetching` | prefetching on | Disable overlapping the next layers' reads with the current compute. |
| `--layer-shards-path` `--delete-original` `--hf-token` | as above | As above. |

Every knob also takes an environment variable, which is the easiest way to set one for a run you did
not launch yourself:

```bash
ROCKETLLM_RESERVE_BYTES=2147483648 rocketllm serve --model ...
ROCKETLLM_HOST_CACHE_BYTES=0       rocketllm serve --model ...   # no host tier at all
ROCKETLLM_IO_WORKERS=1             rocketllm serve --model ...
```

One environment variable is not a RocketLLM knob but matters as much as any of them. Set it before
the first CUDA allocation; `rocketllm doctor` warns when it is missing:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

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

Add `"stream": true` for server-sent events, terminated by `data: [DONE]`.

### Pointing a client at it

Any OpenAI-compatible client works by setting the base URL. There is no authentication, so the API
key is ignored — pass any non-empty string, because most clients refuse to start without one.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-used")

reply = client.chat.completions.create(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",   # or whatever --served-model-name reports
    messages=[{"role": "user", "content": "Name three primary colours."}],
    max_tokens=40,
)
print(reply.choices[0].message.content)
```

The same two settings work anywhere the convention is followed:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=not-used
```

`GET /v1/models` reports the id to use. If you did not pass `--served-model-name`, that is the model
directory's name rather than the repo id you typed.

Two behaviours worth knowing before you wire up a client. Requests are served **one at a time**, so a
client with a short timeout and several parallel calls will time out waiting in the queue rather than
failing outright — each response carries its position in `x-rocketllm-queue-position`. And the first
request after startup pays for the first streaming pass over the weights, which on a storage-bound
machine can be much longer than every request after it.

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

### Tool calling

Tool calls are returned in OpenAI's format, streaming and non-streaming, with `finish_reason` set to
`tool_calls`. Send `tools` on the request and the definitions are rendered into the prompt through
the model's own chat template; the reply is then read back with a parser chosen from what that
template emits.

| Family | Raw syntax it emits |
| --- | --- |
| `hermes` | `<tool_call>{"name": …, "arguments": {…}}</tool_call>` — Hermes, Qwen, and the many fine-tunes on that template |
| `mistral` | `[TOOL_CALLS] [{"name": …, "arguments": {…}}]` — one marker, a JSON array |
| `llama` | `{"name": …, "parameters": {…}}`, optionally after `<|python_tag|>`, several separated by `;` |
| `deepseek` | DeepSeek's full-width delimiters, with the function name outside the JSON |
| `generic` | Fallback: a fenced or bare JSON object with a name — used when the template says nothing |

Detection reads the chat template rather than the model's name, so a fine-tune that inherits its
parent's template is recognised without being listed anywhere. `--tool-parser <family>` overrides it.
`GET /health` reports the family in use, which is the first thing to check if tool calls come back as
prose.

Only a request that supplied `tools` is parsed for them — a model that writes `<tool_call>` in an
ordinary answer is quoting, not calling. `tool_choice` accepts `none` (tools withheld entirely),
`auto`, `required`, and a named function; the last two bias the prompt but are not enforced, because
nothing here constrains decoding, and the server says so rather than implying a guarantee.

Adding a family takes a subclass of `ToolCallParser` in
[toolcalls.py](rocketllm/server/toolcalls.py) and a line in its `PARSERS` registry — set the start
markers, say how the template is recognised, and return where each call's JSON begins. Reading the
name and streaming the arguments is shared. The tests pick up a new family automatically once it has
a captured sample in the corpus at the top of [tests/test_toolcalls.py](tests/test_toolcalls.py).

### Prefix caching

An agentic client resends the whole conversation every turn, so turn three's prompt is turn two's
prompt plus everything since. The server hashes the token sequence in blocks, stores checkpoints of
the KV cache against those hashes, and on a hit restores one and prefills only the tail. It handles
both cache layouts, including the int4 one — a checkpoint is captured while the cache is at exactly
that length, never reconstructed by truncating a longer one, because the boundary between quantized
blocks and the fp16 residual window is a function of the length and getting it wrong produces
slightly wrong output rather than an error.

**It is off by default unless the weights fit resident, and that is not conservatism.** A prefill is
one streaming pass: every weight crosses the link once whether the prompt is forty tokens or four
thousand. Where the model does not fit, that pass is the cost and skipping prompt tokens does not
skip it — measured on an RTX 3090 with the weight cache squeezed to 1GB, three turns of an
1800-token conversation ran **8.6% slower** in prefill with reuse on, having genuinely reused 3741
of 5634 prompt tokens. Where the weights are resident the pass is free and prefill is compute, and
the same conversation ran **14–18% faster** per reused turn. `--prefix-cache auto` follows that
measurement; `on` and `off` force it.

```bash
python tests/bench_streaming.py --model <model> --conversation 3
```

replays a multi-turn conversation with reuse on and off and prints the measured prefill time of
each turn.

Every `rocketllm serve` flag is listed under [Configuration](#configuration) above, and every one
defaults to what `rocketllm profile` measured on your machine. `rocketllm serve --help` prints the
same list with the current defaults filled in.

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

## Performance expectations

**RocketLLM cannot tell you how fast it will be on your machine, and neither can anyone else's
numbers.** What it can tell you is which regime you are in, which is the thing that actually decides
the answer.

Time per token is approximately

```
sum over tiers of:  (bytes that tier has to serve) / (that tier's measured bandwidth)
```

with the tiers, fastest to slowest: VRAM → host RAM across the link → NVMe → SATA SSD → spinning
disk. Every quantity in that formula is measured on your hardware, and the spread between machines is
orders of magnitude rather than percentages.

What decides the regime is **how much of the model fits where**:

- **Fits resident.** The weights sit on the device and are read at device memory bandwidth, which is
  hundreds of GB/s. You are compute- and bandwidth-bound, and RocketLLM is doing almost nothing —
  this is the case where a normal loader would have worked too.
- **Fits in host RAM but not VRAM.** Every weight crosses the link once per token, at single-digit to
  low-double-digit GB/s. Slower, entirely usable.
- **Exceeds VRAM and host RAM.** Every weight is read from storage on every token. This is what
  RocketLLM exists for, and it is bounded by your disk. On a fast NVMe that is seconds per token for
  a large model; on a rotational disk it is far worse and no tuning recovers it.

Speculative decoding, prefix caching and expert residency all attack the same thing: fewer bytes
moved, or the same bytes served from a faster tier, or one streaming pass amortized over more tokens.
None of them changes the tier you are bound by.

### Two example measurements

Clearly labelled as **examples from one machine, not as a promise or a specification**. They exist to
show the size of the gap between regimes, and both were measured with `tests/bench_streaming.py` on
the *same card and the same model* — the only difference is how much of the model was allowed to stay
resident. Your machine will produce different numbers.

Hardware profile for both rows: NVIDIA GeForce RTX 3090, 24GB VRAM, CC 8.6, 15.9GB host RAM,
measured 776 GB/s device memory, 10.5 GB/s pinned host→device, 894 MB/s storage read.
Model: TinyLlama-1.1B-Chat-v1.0, bf16, 16 new tokens. Profile fingerprint `51d86a604e6f4a5e`.

| Regime | Bytes/token (device / host / storage) | Device cache hit rate | Decode | Peak VRAM |
| --- | --- | --- | --- | --- |
| Weights fit resident (default) | 128 B / 131 MB / 131 MB | 94% | 20.7 tok/s | 2.1 GB |
| Cache squeezed to nothing, no host tier | 128 B / 2.20 GB / 2.20 GB | 0% | 0.94 tok/s | 219 MB |

The same card, the same weights, and a **22× difference** — produced entirely by whether the engine
was allowed to keep the model resident. That ratio is the whole point of the tiered cache, and it is
also why a tok/s figure quoted without its hardware profile and its residency is meaningless.

The second row was forced with `--vram-reserve 24696061952 --host-cache-gb 0`, which is how to
emulate a machine much smaller than the one you have. To see where *your* machine would land before
downloading anything:

```bash
rocketllm doctor --model-size 70B --weight-bits 4
```

That prints the projected per-token cost from your measured bandwidths, broken down by tier, with any
tier it could not measure reported as unavailable rather than guessed.

## Community benchmarks

The headline number is **bytes read per token, broken down by memory tier** — tokens per second on
its own says more about the machine than about the engine, and two machines with identical tok/s can
be in completely different regimes.

This table is for results from hardware the author does not own, which is most hardware. A slow
result is as useful as a fast one, and one where something degraded unexpectedly is the most useful
of all. See [docs/HARDWARE.md](docs/HARDWARE.md#contributing-a-result-from-your-machine) for how to
produce a row.

| Hardware profile | VRAM | Host RAM | Weight storage | Model | Quant | Bytes/token (device / host / storage) | Decode tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 3090, CC 8.6, `51d86a604e6f4a5e` | 24 GB | 15.9 GB | NVMe, 894 MB/s measured | TinyLlama-1.1B | bf16 | 128 B / 131 MB / 131 MB | 20.7 |
| _awaiting community submissions_ | | | | | | | |

The "hardware profile" column is not optional: it is the fingerprint `rocketllm doctor` prints, plus
enough of the machine to read the row. A number without the machine it came from cannot be compared
to anything.

## Documentation

| Document | What is in it |
| --- | --- |
| [docs/HARDWARE.md](docs/HARDWARE.md) | Support tiers, what degrades on what and what it costs, how to read `rocketllm doctor`, how to contribute a benchmark result, how to test hardware you do not own. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the streaming engine works: the hook model, the hardware profile and how every knob derives from it, the tiered cache and why dense layers are deliberately not LRU, the MoE expert paths, and the quantization interface. Enough to add a new quantization backend or a new device backend. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to run the suite, what a change has to prove before it lands, and what to put in a bug report. |

## Credits

RocketLLM began as a fork of [AirLLM](https://github.com/lyogavin/airllm) by Gavin Li and the AirLLM
contributors, which originated the layer-streaming approach this engine is built on. RocketLLM has
since been substantially rewritten, and is an independent project — not affiliated with or endorsed
by AirLLM or its authors.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Attribution is binding: [NOTICE](NOTICE) must be retained and reproduced in any distribution or
derivative work, as required by Section 4(d) of the license.
