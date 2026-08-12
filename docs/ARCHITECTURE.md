# Architecture

This document is for someone changing RocketLLM rather than using it. It covers how the streaming
engine works, where every tuning number comes from, why the cache policies are what they are, and how
to add a quantization backend or a device backend without touching anything else.

[docs/HARDWARE.md](HARDWARE.md) is the user-facing companion: support tiers, degradations, and how to
read `rocketllm doctor`.

## The one-paragraph version

The model is built on PyTorch's `meta` device via accelerate's `init_empty_weights`, so it exists as
a module tree with no parameter data and costs no memory. Forward pre-hooks on each big module bring
that module's weights from wherever they currently live onto the device just before it runs; post-
hooks release them. `transformers` still owns the forward pass, the attention implementation, the KV
cache and generation. Everything RocketLLM does happens *around* the model, never inside it.

That is why a newly released architecture generally works with no code change: RocketLLM never
learned what a decoder layer computes, only that it is a module with weights that can be moved.

```
  checkpoint on disk
        │  split once into per-layer shards (utils.split_and_save_layers)
        ▼
  per-layer safetensors shards
        │  LayerLoader: coalesced byte-range reads on N io workers
        ▼
  pooled host staging buffers (pinned where the backend allows)
        │  WeightTransfer: one big copy per layer on a dedicated stream
        ▼
  TieredWeightCache ──► device tier ──► set_module_tensor_to_device ──► the module runs
                   └──► host tier
                   └──► storage (re-read)
```

## Layout

```
rocketllm/
  hw/        profile.py   HardwareProfile: probe the machine, derive every tuning knob
             caps.py      capability queries + the device abstraction (cuda/rocm/mps/xpu/cpu)
             doctor.py    the diagnosis report; no measurement of its own
  memory/    budget.py    VramBudget: continuous free-device-memory measurement, with hysteresis
             cache.py     TieredWeightCache: device / host / storage
             placement.py what to pin, ranked by bytes-saved-per-resident-byte
  streaming/ loader.py    coalesced reads into pooled host buffers, N io workers
             staging.py   the host buffer pool
             transfer.py  dedicated copy stream, one transfer per layer, + synchronous fallback
  quant/     registry.py  the format-agnostic PackedWeight / QuantBackend interface
             safetensors_quant.py  AWQ / GPTQ / compressed-tensors / MXFP4 / bnb-prequantized
             kv_cache.py  KIVI-style int4 KV cache
  moe/       detect.py    structural expert-container detection (ModuleList and fused 3D)
             expert_cache.py  hot-expert LFU residency + intra-layer parallel fetch
             router.py    router interception
  spec/      draft.py     speculative decoding
  server/    app.py protocol.py toolcalls.py prefix_cache.py
  base.py    RocketModel: the load sequence and the hooks
```

## The load sequence, and why it is ordered

`RocketModel.__init__` is one long ordered sequence, and several steps are where they are for reasons
that are not obvious. Changing the order is the easiest way to break this engine subtly.

1. **`configure_allocator(device)` — first, before anything touches the device.** The CUDA caching
   allocator reads `PYTORCH_CUDA_ALLOC_CONF` exactly once, when the context is created. Asking for
   expandable segments after the hardware probe has already allocated something is asking too late,
   and the request silently does nothing. Whether it took effect is *measured*, not assumed.
2. **Resolve the device.** Queried, never assumed. `device=None` means "the fastest backend this
   machine actually has".
3. **Split the checkpoint** into per-layer shards, once, next to the model cache.
4. **Resolve the compute dtype** through `DeviceCaps.select_compute_dtype`, which degrades the
   checkpoint's request to what the device can really do and announces the degradation.
5. **`HardwareProfile.load_or_probe`** — cached by hardware fingerprint, so this is a one-off cost.
6. **Build the model on `meta`**, then install streaming hooks.
7. **`_build_cache`** — sizes the cache from the profile *and* from a live budget reading.
8. **`_setup_speculation`** last, because a draft model is resident by definition and takes device
   memory away from the cache that was just sized. Registering its bytes republishes the budget
   immediately, so the cache shrinks to fit rather than discovering the shortfall on the first token.

## The hardware profile

Everything in the engine that could have been a constant is instead a value from
`rocketllm/hw/profile.py`. There is no reference machine, so a constant that was right on one box is
a bug on every other one.

Two kinds of number live there, and the distinction is load-bearing:

- **Measured or queried** — device memory, bandwidths, core counts, dtype support, allocator
  fragmentation. Never assumed. If a backend will not report one it stays `None`, and every consumer
  must treat `None` as *unknown*, never as zero.
- **Policy factors** — the dimensionless fractions in `Policy`, such as what share of usable device
  memory a prefetch window may claim. These are design choices, not properties of the machine. They
  live in one dataclass, are carried in the profile, printed with the value they produced, and are
  overridable. Calling them "measured" would be a lie; scattering them through the code would be
  worse.

### How a knob is derived

Every derived value records the formula that produced it and the inputs that went in:

```python
self._set("reserve_bytes", value,
          "min(max(measured_workspace, total_device * measured_fragmentation_ratio), "
          "total_device * reserve_ceiling_fraction)",
          {"total_device_bytes": total, "measured_workspace_bytes": workspace, ...},
          overrides)
```

`rocketllm profile` prints that verbatim, which is usually enough to see why a number came out the
way it did without reading this file.

**The resolution order is: explicit argument → environment variable → derived value.** `_knob()` on
`RocketModel` implements it, and every new knob must support all three.

### Adding a knob

1. Add the policy factor to `Policy` if it needs one, with a comment saying why that value.
2. Add a `_derive_*` method (or extend one), calling `self._set(name, value, formula, inputs,
   overrides)`. The formula string is not decoration — it is what a bug report will be read against.
3. Add the name to `_OVERRIDABLE` with its parser, so `ROCKETLLM_<NAME>` works.
4. Bump `SCHEMA_VERSION`. Profiles are cached and replayed by hardware fingerprint; a stale cache
   would otherwise keep supplying a knob that no longer means what it did.
5. Read it through `self._knob("name", fallback)`, never directly.

### The fingerprint

Cached profiles are keyed by a hash of the machine's identity — backend, device name, memory totals,
CPU count, torch version, weights path. Swapping a card, moving machines or upgrading torch simply
stops matching and a fresh probe runs. Software versions are in there deliberately: a torch upgrade
can change allocator behaviour and kernel availability, which changes what would be derived.

Measurements are reused on a cache hit; **derivations are always recomputed**, because they depend on
live free memory and on overrides.

## The tiered cache

`TieredWeightCache` holds three tiers — device, host, storage — and the entries in it are *packed
bytes*. Nothing is stored expanded. Dequantization happens into a small reusable scratch buffer
immediately before the layer runs and is freed after, so all sizing and placement arithmetic is in
packed bytes throughout.

Entries are refcounted. An entry that is in use is never evicted; a pinned entry is never evicted at
all.

### Why dense layers are not LRU

**This is the single most important design decision in the cache, and it is the one most likely to be
"simplified" by someone who has not hit the failure.**

Decoder layers are accessed *cyclically*: 0, 1, 2, … L, then 0 again for the next token. Consider a
cache that holds K layers out of L, with K < L, under LRU. After the scan passes layer K, the least
recently used entry is layer 0 — which is precisely the layer the next token will need first. LRU
evicts exactly what is about to be requested, on every step, forever. The hit rate is not merely
poor; it is approximately **zero**, and it gets no better with a larger cache until the cache holds
the entire model.

This is the classic cyclic-scan pathology, and no recency heuristic escapes it, because recency is
anti-correlated with next use under a cyclic access pattern.

So dense modules use:

- **a statically pinned subset**, chosen once against the pin budget and held for the whole run, plus
- **a FIFO prefetch window** of the next few layers, which moves forward with the scan.

The pinned subset gives a hit rate proportional to how much of the model fits — which is the correct
behaviour, and what LRU fails to deliver. The window covers the layers the scan is about to reach.

**MoE experts are the exception, and the asymmetry is deliberate.** Expert popularity is skewed and
self-predicting: an expert that has been hot is likely to stay hot, because routing is a property of
the model and the data rather than of position in a scan. So experts *do* use LFU-with-aging. Aging
is what stops the counts freezing around whatever was hot at the start of a long generation.

If you take one thing from this document: the two policies are different because the two access
patterns are different, and unifying them will silently destroy the dense hit rate.

### The device memory budget

```
budget = free_from_driver
       + (memory_reserved() - memory_allocated())
       - reserve
       + bytes the cache already holds
       ... floored at zero ONCE, at the very end
```

Each term earns its place:

- `mem_get_info()` alone **under-reports**, because the caching allocator sits on blocks it has
  already freed and the driver still counts those as in use. Adding the difference back is the
  difference between a cache that fills the card and one that gives up with room to spare.
- The cache's own bytes are not free, but they are *reclaimable by the thing being sized*, so they
  belong inside the budget.
- **Floor once, at the end.** Flooring before adding the cache's own bytes degenerates the answer to
  "however much you are already holding" everywhere below the reserve. The cache can then only
  ratchet *down* — every eviction lowers its own ceiling — and memory freed mid-generation becomes
  invisible.

`reserve` comes from the profile, built from the allocator's measured fragmentation ratio and
workspace high-water mark. It is not a constant.

### Hysteresis

The budget is measured, so it carries the allocator's noise. Acting on that noise means evicting and
refetching on a reading that will be taken back a sample later — a whole streaming pass to learn
nothing. So a change must exceed a band and persist for several consecutive samples before it is
published.

**The band is a *share* of the budget in play, never a byte count sized off the whole card.** Sized
off the card it comes out larger than the budget it is meant to damp — measured at 1693MB against a
live budget of 507MB — and then no change of any size can ever be published, so the pin plan never
moves. A share means the same thing on a 4GB card and a 192GB one.

### Order of operations when the budget moves

**Adopt the new pin plan *before* resizing the cache.** Pinned entries are never evicted, so resizing
first cannot free what the smaller budget has just unpinned. This is a one-line ordering constraint
with a very confusing failure mode if reversed.

### Pin planning

`placement.py` ranks candidates by **bytes saved per resident byte** — accesses per token divided by
packed size. A weight read twice per token is worth twice as much resident; a weight half the size is
worth twice as much per byte it occupies. The fill is greedy over that ranking, which is a knapsack
approximation and good precisely because the ranking is by value density. Candidates that do not fit
are skipped rather than terminating the fill, so a small high-value weight further down still gets
room a huge one could not use.

Classes are absolute and ordered: always-on modules, then shared experts, then routed experts. No
expert displaces a module that every token reads.

The pin budget is what is left of usable device memory *after* the prefetch window is committed —
the window is not optional, because a layer that is not resident still has to land somewhere before
it can run. On a small device that subtraction legitimately yields zero, and **zero must work**: it
means pure streaming.

## MoE

### Detection is structural

`moe/detect.py` decides whether a layer is a mixture from the module tree and the checkpoint's tensor
shapes — never from an architecture name. That is what makes unreleased models work without code
changes.

Two failure directions, and they are not symmetric:

- **Missing a real mixture costs speed.** The layer streams whole, exactly as before, output
  unaffected. Recoverable, and visible in the benchmark.
- **Claiming a mixture that is not one, or reading the wrong experts, costs correctness.** Nothing
  raises; the model keeps generating, slightly wrong.

Every ambiguous case therefore resolves toward *not* a mixture.

### Two layouts, both required

- **`LAYOUT_MODULE_LIST`** — an `nn.ModuleList` of expert modules. Each expert is its own cache
  entry and streams on its own.
- **`LAYOUT_FUSED`** — every expert batched into one tensor per projection, e.g.
  `experts.gate_up_proj` shaped `[E, ...]`. The routed rows are read with
  `safe_open(...).get_slice(key)[e:e+1]`, so one expert costs *its own* bytes rather than the whole
  layer's. Reading the layer and slicing afterwards would defeat the entire feature.

### There is no cross-layer router lookahead

Layer L's router runs *inside* layer L. Layer L+1's experts are unknowable until L completes. Do not
design around a lookahead that cannot exist.

What does work, and is implemented:

- hot-expert LFU residency across tokens,
- parallel fetch of the top-k *within* a layer, once that layer's router has fired,
- pinning shared/always-on experts, which run for every token by construction.

## The quantization interface

RocketLLM **imports** pre-quantized checkpoints; it never quantizes a model itself.

Two objects carry the whole abstraction:

- **`PackedWeight`** — one logical weight, described without holding its data: which checkpoint
  tensors it spans, what it costs packed and expanded, and how to place it. The cache sizes layers
  from safetensors headers alone, without reading a byte.
- **`QuantBackend`** — the per-format knowledge behind that. Subclasses answer how a format lays a
  logical weight across checkpoint tensors, how big it is, and *which device capability* decides
  whether it can be computed on as stored.

### The split that matters

**A format decides which capability gets asked. The device decides the answer.**

Whether a weight stays packed through the matmul is a property of the *machine*, not of the file: the
same AWQ checkpoint computes from the packed form on a card with a fused kernel and must be expanded
into scratch on one without. So `needs_scratch` asks `rocketllm.hw.caps` every time, and two machines
reading the same file will correctly disagree.

Never gate on the format name, and never bake a device assumption into a backend.

### Adding a quantization backend

1. Subclass `QuantBackend` in `rocketllm/quant/safetensors_quant.py` (or a new module imported from
   `_load_backends`).
2. Set `format` and `quant_methods` — the `quant_method` values from a checkpoint's
   `quantization_config` that should select it. Override `matches()` if selection needs more than a
   name.
3. Implement the layout questions:
   - `logical_name(tensor_name, known)` — map a checkpoint tensor back to the logical weight it
     belongs to (strip `_scale`, `_zero_point`, `_g_idx`, and friends; see `COMPANION_SUFFIXES`).
   - `is_payload` / `is_consumed` — which tensor carries the values, and which are companions.
   - `bits`, `values_per_item`, `logical_shape` — enough to compute packed and expanded sizes.
4. Implement `needs_scratch(weight)` **as a capability query**, not as a constant.
5. Override `place()` only if placement genuinely differs — for most formats it is the same
   `set_module_tensor_to_device` call, or the quantizer's own reconstruction.
6. Provide `example_config()`. The decision table builds one backend per format without a checkpoint
   to hand, so `rocketllm doctor` can report what *would* happen here.
7. Implement `decision()` so the format explains itself in one line: which path, and which query
   chose it.
8. Register it with `@register_backend`. **First match wins, so register specific backends before
   general ones.**
9. Add a case to `tests/test_quant_registry.py`. Sizing must be right in packed bytes, and
   `needs_scratch` must flip with the mocked capability rather than with the format.

An unrecognised `quant_method` is not a failure: `HfQuantizerBackend` delegates to whatever quantizer
transformers wired up for it.

## Adding a device backend

The device abstraction is `DeviceCaps` in `rocketllm/hw/caps.py`, plus one subclass per backend. It
covers exactly what the streaming path needs, so the engine calls the same handful of methods
everywhere and never branches on a backend itself.

1. **Add the query functions.** Extend `available_backends()`, `resolve_device()`, `backend_of()`,
   `device_memory()`, `synchronize()`. Every one answers by *asking the backend or attempting the
   operation* — never by recognising a device name. Return `None` when the backend cannot say;
   `None` means unknown and callers must not fold it into `False`.
2. **Subclass `DeviceCaps`.** Set `backend` and `tier`. Derive `tier` from a capability where the
   backend has one to derive it from (see `CudaCaps.tier`, which reads the compute capability).
3. **Override only what the backend can do natively.** Everything else inherits a fallback that is
   slower and still correct:

   | Method | Base behaviour if you do not override it |
   | --- | --- |
   | `memory(reserve_bytes)` | Whatever the backend calls free, marked `estimated=True` |
   | `can_pin_memory` | `False` → pageable staging buffers |
   | `has_async_streams` | `False` → `_SyncStream`, a no-op stand-in with the same surface |
   | `copy_stream()` / `event()` | The synchronous stand-ins |
   | `empty_cache()` | `gc.collect()` |
   | `fused_4bit_plan()` | `dequant_to_scratch` |
   | `select_compute_dtype()` | bf16 → fp16 (loudly) → fp32 |

4. **Register it in `_BACKEND_CLASSES`.**
5. **Announce every degradation once.** `announce_degradations()` is called when the device is first
   resolved. Use `announce_once(key, message, level)` — never log per layer, because a streaming run
   touches every module hundreds of times per token.
6. **Add the backend to `tests/test_hw_caps.py`'s `BACKENDS` table.** The suite then asserts, with
   every query mocked, that each gate decides, each fallback is reachable and produces something
   usable, and nothing raises because a feature is absent.
7. **Add an emulated case to `tests/test_portability.py`** if the backend has a memory model worth
   sweeping.

The rule that makes this work: **a missing hardware feature is a slower path, never an error.**
Nothing in `caps.py` raises because an optional accelerator feature is absent.

## Where the invariants are tested

| Invariant | Test |
| --- | --- |
| Dense layers do not use LRU; cyclic scan keeps its hit rate | `tests/test_cache_policy.py` |
| Budget arithmetic, floor-once, hysteresis as a share | `tests/test_vram_budget.py` |
| Every knob derives sensibly across four mocked machines | `tests/test_hw_profile.py` |
| Every gate decides and every fallback works, per backend | `tests/test_hw_caps.py` |
| Format sizing in packed bytes; `needs_scratch` follows the device | `tests/test_quant_registry.py` |
| Expert layout detected structurally, ambiguity resolves to "not a mixture" | `tests/test_moe_detect.py` |
| Streamed logits identical to a full load, dense and MoE | `tests/test_moe_streaming.py` |
| Streamed generation identical to a full load, on CPU | `tests/test_cpu_generation.py` |
| Correct output and designed degradation across emulated devices | `tests/test_portability.py` |
| The package imports with every optional dependency absent | `tests/test_optional_imports.py` |

The gate that outranks all of them is `tests/test_streaming_gpu.py --compare`, which must report
`MATCH`. A fast engine that produces the wrong tokens is worth nothing.

## Things that look like improvements and are not

- **Unifying the dense and expert cache policies.** See above; it zeroes the dense hit rate.
- **LRU anywhere on the cyclic scan.** Same reason.
- **Storing expanded weights in the cache.** All the sizing arithmetic is in packed bytes, and
  expansion multiplies residency by four or more for no gain — the scratch buffer already exists.
- **Sizing the hysteresis band off total device memory.** It then exceeds the budget it damps.
- **Resizing the cache before adopting a new pin plan.** Pinned entries cannot be evicted.
- **Designing around cross-layer router lookahead.** It cannot exist.
- **Continuous batching.** Two sequences sharing this engine evict each other's layers and both run
  slower than either alone. Deliberately out of scope.
- **Reading a whole fused expert tensor and slicing after.** That is the cost the fused path exists
  to avoid.
