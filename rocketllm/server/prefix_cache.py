"""Reusing the KV cache of a prefix a client has already sent.

An agentic client resends the whole conversation every turn: turn three's prompt is turn two's
prompt plus turn two's answer plus a tool result. Without reuse each turn re-prefills thousands of
tokens it prefilled last turn, and on a streaming engine that is the single most expensive thing a
request can do -- prefill moves every weight through the device once, so re-prefilling 4000 tokens
costs a full streaming pass that buys nothing.

How it works: the token sequence is hashed in blocks, each block's hash chained onto the one before
it, so a shared prefix produces a shared chain of hashes and the deepest matching link is the
longest reusable prefix. Checkpoints of the cache are stored against those chain hashes. On a hit
the checkpoint is restored and only the tail is prefilled.

LRU HERE, UNLIKE THE LAYER CACHE. rocketllm/memory/cache.py deliberately refuses LRU for decoder
layers, because they are read cyclically -- 0..L, 0..L -- and on a cyclic scan larger than the cache
LRU evicts precisely the entry needed next, giving a ~0% hit rate. Prefixes are the opposite: a
conversation is touched repeatedly over a few minutes and then never again, which is recency, which
is what LRU is for. The two policies differ because the access patterns differ, not because one of
them has been left unmodernised. Do not "unify" them.

**The int4 boundary is the part that goes wrong quietly.** The quantized cache holds whole groups of
tokens as int4 blocks plus an fp16 residual window of the most recent ones. Which tokens sit on
which side is a function of the length alone:

    quantized(n) = max(0, ((n - residual_length) // group_size) * group_size)

That identity is why a checkpoint can be exact -- it holds however the tokens arrived, one prefill or
a thousand single steps -- and it is also the trap. Restore a cache at a length whose split does not
match, and the tokens near the boundary are quantized where a real prefill would have kept them
exact. Attention still runs, nothing raises, and the output is slightly wrong for the rest of the
conversation. So checkpoints are captured *while the live cache is at that exact length* and
restored at that same length, never reconstructed by truncating a longer one: truncation would have
to dequantize tokens back into the residual window, and a dequantized token is not the token a real
prefill would have had there. tests/test_prefix_cache.py checks the restored split against a
from-scratch prefill directly, because that is the failure this design exists to avoid.
"""
import hashlib
import logging
import os
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..hw import caps

log = logging.getLogger("rocketllm.server")

#: Bumped when the stored layout changes, so a spilled checkpoint written by an older build is
#: ignored rather than restored into a cache that no longer means the same thing by it.
FORMAT_VERSION = 1

KIND_QUANTIZED = "quantized"
KIND_DYNAMIC = "dynamic"


# ---- hashing --------------------------------------------------------------------------------

def _token_bytes(tokens):
    # Fixed-width little-endian, so token ids cannot run together into a different sequence that
    # hashes the same.
    return b"".join(int(token).to_bytes(4, "little") for token in tokens)


def chain_hashes(tokens, block_size, seed=b""):
    """The hash of the first k whole blocks, for every k. Index 0 is the empty prefix.

    Chained rather than per-block so a match at depth k means the first k*block_size tokens are
    identical, not merely that block k happens to be the same. That is what lets a lookup take the
    deepest matching link without comparing a single token.

    Truncated to 128 bits. A collision would restore the wrong cache and produce plausible wrong
    output, which is the worst failure mode here, so the margin is deliberately enormous rather
    than merely sufficient: at 2**64 stored prefixes the odds are still around 2**-64.
    """
    running = hashlib.blake2b(seed, digest_size=16).digest()
    hashes = [running.hex()]
    for index in range(len(tokens) // block_size):
        block = tokens[index * block_size:(index + 1) * block_size]
        digest = hashlib.blake2b(running, digest_size=16)
        digest.update(_token_bytes(block))
        running = digest.digest()
        hashes.append(running.hex())
    return hashes


def tail_hash(tokens):
    """The hash of the partial block hanging off the end of a prefix.

    A checkpoint's length is almost never a multiple of the block size -- prefill lands wherever the
    prompt ends, and one prefill pass moves every weight through the device, so chunking it to land
    on a boundary would cost a whole streaming pass per block. So an entry is keyed by the block
    chain up to the last whole block PLUS the hash of the tokens after it, which makes any length
    indexable while keeping the chain the thing that finds it.
    """
    return hashlib.blake2b(_token_bytes(tokens), digest_size=16).hexdigest()


def namespace_seed(*parts):
    """A seed binding a prefix to the model that produced it.

    Two models produce different KV for the same tokens, so a prefix cache shared between them
    would restore a cache of the wrong shape at best and wrong numbers at worst. The seed makes
    that impossible rather than merely unlikely.
    """
    joined = "\x1f".join(str(part) for part in parts)
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).digest()


# ---- what a checkpoint holds ------------------------------------------------------------------

def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _move_block(block, device):
    """One quantized block, on another device. Immutable, so this is a copy and never a view."""
    import dataclasses

    return dataclasses.replace(
        block,
        packed=block.packed.to(device, copy=False),
        scale=block.scale.to(device, copy=False),
        zero=block.zero.to(device, copy=False))


class _BlockStore:
    """Host-side copies of quantized blocks, shared between the checkpoints that reference them.

    Sharing is safe because the int4 cache is append-only: once a group of tokens has been encoded
    it is never rewritten. So the checkpoints of one conversation hold overlapping prefixes of one
    growing list, and copying each of them in full would multiply a 128MB context by the number of
    checkpoints. Refcounted rather than garbage-collected so the size cap can be enforced against a
    real number.
    """

    def __init__(self):
        self._payloads = {}
        self._refs = {}
        self._sizes = {}
        self._next = 0
        self.bytes = 0

    def put(self, payload, nbytes):
        key = self._next
        self._next += 1
        self._payloads[key] = payload
        self._refs[key] = 0
        self._sizes[key] = nbytes
        self.bytes += nbytes
        return key

    def get(self, key):
        return self._payloads[key]

    def acquire(self, keys):
        for key in keys:
            self._refs[key] += 1

    def release(self, keys):
        for key in keys:
            self._refs[key] -= 1
            if self._refs[key] <= 0:
                self.bytes -= self._sizes.pop(key, 0)
                self._payloads.pop(key, None)
                self._refs.pop(key, None)


@dataclass
class Checkpoint:
    """A cache, exactly as it stood after `length` tokens."""

    length: int
    kind: str
    #: Per layer, for the quantized layout: (key block store keys, value block store keys).
    blocks: tuple = ()
    #: Per layer: the fp16 tensors. The residual window for the quantized layout, the whole cache
    #: for the dynamic one.
    residual: tuple = ()
    #: What the layout was configured with, so a restore into a differently configured cache is
    #: refused rather than silently producing a different split.
    group_size: int = 0
    residual_length: int = 0
    resident_bytes: int = 0

    def block_keys(self):
        return [key for layer in self.blocks for side in layer for key in side]


# ---- capture and restore -----------------------------------------------------------------------

def capture(cache, store, memo=None):
    """Snapshot the live cache exactly as it stands. Returns a Checkpoint, or None if unsupported.

    `memo` maps the identity of a device-side block to the store key of its host copy, so the
    blocks a previous checkpoint of the same generation already copied are not copied again.
    """
    memo = memo if memo is not None else {}
    from ..quant.kv_cache import QuantizedKVCache

    if isinstance(cache, QuantizedKVCache):
        return _capture_quantized(cache, store, memo)
    if _is_dynamic(cache):
        return _capture_dynamic(cache)
    return None


def _is_dynamic(cache):
    from transformers.cache_utils import DynamicCache

    return isinstance(cache, DynamicCache)


def _capture_quantized(cache, store, memo):
    layers = []
    residual = []
    for layer_idx in range(len(cache.key_cache)):
        key_blocks, value_blocks = cache._blocks[layer_idx]
        keys = tuple(_hold(block, store, memo) for block in key_blocks)
        values = tuple(_hold(block, store, memo) for block in value_blocks)
        layers.append((keys, values))
        residual.append((cache.key_cache[layer_idx].detach().to("cpu", copy=True),
                         cache.value_cache[layer_idx].detach().to("cpu", copy=True)))
    checkpoint = Checkpoint(
        length=cache.get_seq_length(), kind=KIND_QUANTIZED, blocks=tuple(layers),
        residual=tuple(residual), group_size=cache.config.group_size,
        residual_length=cache.config.residual_length,
        resident_bytes=sum(_tensor_bytes(k) + _tensor_bytes(v) for k, v in residual))
    return checkpoint


def _hold(block, store, memo):
    key = memo.get(id(block))
    if key is None:
        host = _move_block(block, "cpu")
        key = store.put(host, host.nbytes)
        memo[id(block)] = key
    return key


def _capture_dynamic(cache):
    residual = tuple((cache.key_cache[i].detach().to("cpu", copy=True),
                      cache.value_cache[i].detach().to("cpu", copy=True))
                     for i in range(len(cache.key_cache)))
    return Checkpoint(
        length=cache.get_seq_length(), kind=KIND_DYNAMIC, residual=residual,
        resident_bytes=sum(_tensor_bytes(k) + _tensor_bytes(v) for k, v in residual))


def restore(checkpoint, store, device, config=None, memo=None):
    """Rebuild a live cache from a checkpoint. The inverse of `capture`, exactly.

    `memo` is filled with the identity of each device block against the store key it came from, so
    the checkpoints taken during the generation that follows can share those host copies instead of
    copying them straight back.
    """
    if checkpoint.kind == KIND_DYNAMIC:
        return _restore_dynamic(checkpoint, device)
    return _restore_quantized(checkpoint, store, device, config, memo)


def _restore_dynamic(checkpoint, device):
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    cache.key_cache = [key.to(device, copy=True) for key, _ in checkpoint.residual]
    cache.value_cache = [value.to(device, copy=True) for _, value in checkpoint.residual]
    cache._seen_tokens = checkpoint.length
    return cache


def _restore_quantized(checkpoint, store, device, config, memo):
    from ..quant.kv_cache import QuantizedKVCache

    if config is not None and (config.group_size != checkpoint.group_size
                               or config.residual_length != checkpoint.residual_length):
        # The split between quantized and exact tokens is a function of these two numbers. A
        # checkpoint restored under different ones puts the boundary somewhere the running cache
        # would never have put it, and nothing downstream would notice.
        raise ValueError(
            f"this checkpoint was taken with group_size={checkpoint.group_size} and "
            f"residual_length={checkpoint.residual_length}, but the cache is configured for "
            f"group_size={config.group_size} and residual_length={config.residual_length}. "
            f"Restoring it would put the quantized/exact boundary in the wrong place.")

    cache = QuantizedKVCache(config)
    cache._blocks = []
    cache.key_cache = []
    cache.value_cache = []
    for layer_idx, (key_keys, value_keys) in enumerate(checkpoint.blocks):
        key_blocks = [_release_to(store.get(key), device, key, memo) for key in key_keys]
        value_blocks = [_release_to(store.get(key), device, key, memo) for key in value_keys]
        # New lists, not the stored tuples: the cache appends to these as it runs, and appending to
        # something a checkpoint holds would rewrite history the checkpoint promised was frozen.
        cache._blocks.append((key_blocks, value_blocks))
        residual_key, residual_value = checkpoint.residual[layer_idx]
        cache.key_cache.append(residual_key.to(device, copy=True))
        cache.value_cache.append(residual_value.to(device, copy=True))
    cache._seen_tokens = checkpoint.length
    return cache


def _release_to(host_block, device, key, memo):
    block = _move_block(host_block, device)
    if memo is not None:
        memo[id(block)] = key
    return block


# ---- checkpointing caches -------------------------------------------------------------------------

def checkpointing(cache, observer, layers):
    """Return `cache` with `observer` called once per completed step, or the cache untouched.

    Snapshots have to be taken while the cache is at the length they claim -- see the module
    docstring -- so something has to watch it as generation runs. `update` is the only per-step
    entry point a Cache has, and it is called once per layer, so the last layer is the point at
    which every layer is consistent at the new length.
    """
    from ..quant.kv_cache import QuantizedKVCache

    if cache is None or not layers or layers < 1:
        return cache
    if not isinstance(cache, (QuantizedKVCache,)) and not _is_dynamic(cache):
        caps.announce_once(
            "prefix-cache-layout",
            f"prefix caching does not know how to snapshot a {type(cache).__name__}, so this run "
            f"prefills every turn from scratch. Correct, just slower; the int4 and full-precision "
            f"caches are the two it handles.", logging.INFO)
        return cache
    cache.__class__ = _make_checkpointing(type(cache))
    cache._rocketllm_observer = observer
    cache._rocketllm_final_layer = layers - 1
    return cache


def _observe(cache, layer_idx):
    if layer_idx == cache._rocketllm_final_layer:
        cache._rocketllm_observer(cache)


_SUBCLASSES = {}


def _make_checkpointing(base):
    """A watched subclass of whichever cache this is, made once and reused.

    A subclass rather than a wrapper: a wrapper would mean reimplementing the whole Cache surface
    and hoping transformers never does an isinstance check on it. Rebinding __class__ to a subclass
    that adds no fields is layout-compatible, so the object every other part of the stack is
    holding does not change identity or shape.
    """
    made = _SUBCLASSES.get(base)
    if made is None:
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            out = base.update(self, key_states, value_states, layer_idx, cache_kwargs)
            _observe(self, layer_idx)
            return out

        made = type(f"Checkpointing{base.__name__}", (base,), {"update": update})
        _SUBCLASSES[base] = made
    return made


# ---- the cache ------------------------------------------------------------------------------------

@dataclass
class Match:
    """What a lookup found."""

    length: int
    blocks: int
    checkpoint: Checkpoint
    #: Where it came from, for the report: "host" or "storage".
    tier: str = "host"


class _PrefillBaseline:
    """What a full prefill of a given size actually costs here, from requests that did one.

    Time saved cannot be measured on the request that saved it -- a request is served once, one
    way. So the only honest baseline is other requests that DID prefill their whole prompt, at a
    comparable length, on this machine.

    What this replaced was a per-token rate taken from the request's own short prefill and
    multiplied by the tokens it skipped. That is not a baseline, because prefill is parallel across
    tokens and the per-token rate collapses with length. Measured on a 1.1B model with the weights
    resident: 38 tokens took 80ms, or 2.1ms per token, while a full 1878-token prefill took 100ms,
    or 0.05ms per token -- forty times less. The extrapolation claimed 7.92 seconds saved across
    three turns where the measured saving was 0.04.
    """

    def __init__(self, tolerance=0.25, keep=64):
        #: How far a sample's length may sit from the prompt being estimated and still describe it.
        self.tolerance = tolerance
        self.samples = []
        self.keep = keep

    def observe(self, tokens, seconds):
        if tokens <= 0 or seconds is None or seconds <= 0:
            return
        self.samples.append((int(tokens), float(seconds)))
        if len(self.samples) > self.keep:
            self.samples.pop(0)

    def estimate(self, tokens):
        """What a full prefill of this prompt would have cost, or None if nothing comparable ran."""
        if tokens <= 0:
            return None
        near = [seconds for length, seconds in self.samples
                if abs(length - tokens) <= self.tolerance * max(1, tokens)]
        return sum(near) / len(near) if near else None


@dataclass
class PrefixStats:
    lookups: int = 0
    hits: int = 0
    host_hits: int = 0
    storage_hits: int = 0
    tokens_skipped: int = 0
    tokens_prefilled: int = 0
    seconds_saved: float = 0.0
    checkpoints: int = 0
    evictions: int = 0
    spills: int = 0
    spill_errors: int = 0

    def to_dict(self, cache=None):
        total = self.tokens_skipped + self.tokens_prefilled
        report = dict(
            lookups=self.lookups, hits=self.hits, host_hits=self.host_hits,
            storage_hits=self.storage_hits,
            hit_rate=(self.hits / self.lookups) if self.lookups else 0.0,
            tokens_skipped=self.tokens_skipped, tokens_prefilled=self.tokens_prefilled,
            tokens_reused_fraction=(self.tokens_skipped / total) if total else 0.0,
            # Against measured full prefills of comparable prompts on this machine -- see
            # _PrefillBaseline. Zero until one has run, rather than a number from nowhere.
            seconds_saved_vs_measured_baseline=round(self.seconds_saved, 3),
            checkpoints=self.checkpoints, evictions=self.evictions, spills=self.spills,
            spill_errors=self.spill_errors)
        if cache is not None:
            report.update(entries=len(cache._entries), host_bytes=cache.bytes,
                          host_capacity=cache.capacity_bytes,
                          spilled_entries=len(cache._spilled), spilled_bytes=cache.spilled_bytes,
                          spill_capacity=cache.spill_bytes, block_tokens=cache.block_size)
        return report


class PrefixCache:
    """Checkpoints of prefixes people have already paid to prefill.

    Two tiers. The host tier holds live tensors under a byte cap from the hardware profile; what it
    evicts is written to storage, under its own larger cap, because reloading a spilled checkpoint
    is a read and the alternative is a whole streaming pass. Both are LRU -- see the module
    docstring for why that is right here and wrong for the layer cache.
    """

    def __init__(self, block_size=256, capacity_bytes=0, spill_bytes=0, spill_dir=None,
                 seed=b"", enabled=True):
        self.block_size = max(1, int(block_size))
        self.capacity_bytes = max(0, int(capacity_bytes))
        self.spill_bytes = max(0, int(spill_bytes))
        self.spill_dir = Path(spill_dir) if spill_dir else None
        self.seed = seed
        self.enabled = bool(enabled) and self.capacity_bytes > 0
        self.stats = PrefixStats()
        self.baseline = _PrefillBaseline()
        self.store = _BlockStore()
        self._entries = OrderedDict()      # entry key -> Checkpoint, oldest first
        #: block-chain hash -> {length: tail hash}. The lengths hanging off one chain link, so a
        #: lookup knows which partial-block tails are worth confirming.
        self._index = {}
        self._spilled = OrderedDict()      # entry key -> bytes on disk
        self.spilled_bytes = 0
        self._lock = threading.Lock()
        if self.spill_dir is not None:
            try:
                self.spill_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                caps.announce_once(
                    "prefix-spill-unavailable",
                    f"could not create the prefix cache spill directory ({exc}); checkpoints "
                    f"evicted from host memory are dropped instead of being written to storage. "
                    f"Reuse still works, just over a shorter history.", logging.INFO)
                self.spill_dir = None

    @property
    def bytes(self):
        return self.store.bytes + sum(entry.resident_bytes for entry in self._entries.values())

    # -- lookup ----------------------------------------------------------------------------------

    def hashes(self, tokens):
        return chain_hashes(tokens, self.block_size, self.seed)

    def key_for(self, tokens, length, hashes=None):
        """The index key for a checkpoint covering the first `length` of `tokens`."""
        depth = length // self.block_size
        hashes = hashes if hashes is not None else self.hashes(tokens[:length])
        if depth >= len(hashes):
            return None
        return f"{hashes[depth]}:{tail_hash(tokens[depth * self.block_size:length])}"

    def lookup(self, tokens):
        """The longest stored prefix of `tokens`, or None.

        Walks the block chain from the deepest link back. At each link, the entries hanging off it
        differ only in their partial-block tail, so confirming one costs hashing under a block of
        tokens -- and the first confirmed hit at the deepest link is the longest match there is.
        """
        if not self.enabled:
            return None
        hashes = self.hashes(tokens)
        with self._lock:
            self.stats.lookups += 1
            for depth in range(len(hashes) - 1, -1, -1):
                bucket = self._index.get(hashes[depth])
                if not bucket:
                    continue
                base = depth * self.block_size
                # Longest first: at one link the longer tail is strictly more of the prompt reused.
                for length in sorted(bucket, reverse=True):
                    if length > len(tokens):
                        continue
                    if tail_hash(tokens[base:length]) != bucket[length]:
                        continue
                    key = f"{hashes[depth]}:{bucket[length]}"
                    checkpoint = self._entries.get(key)
                    tier = "host"
                    if checkpoint is None:
                        checkpoint = self._load_spilled(key)
                        tier = "storage"
                    if checkpoint is None or checkpoint.length != length:
                        continue
                    if key in self._entries:
                        # A checkpoint read back from storage may not have stayed resident -- on a
                        # tight host budget it can be evicted again by the very fit check that
                        # admitted it. It is still perfectly usable for this request.
                        self._entries.move_to_end(key)
                    self.stats.hits += 1
                    if tier == "host":
                        self.stats.host_hits += 1
                    else:
                        self.stats.storage_hits += 1
                    return Match(length=length, blocks=depth, checkpoint=checkpoint, tier=tier)
        return None

    # -- storing ----------------------------------------------------------------------------------

    def put(self, tokens, checkpoint, hashes=None):
        """Store a checkpoint under the tokens it covers. Returns whether it was kept."""
        if not self.enabled:
            return False
        length = checkpoint.length
        if length <= 0 or length > len(tokens):
            return False
        key = self.key_for(tokens, length, hashes)
        if key is None:
            return False
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return False
            self.store.acquire(checkpoint.block_keys())
            self._entries[key] = checkpoint
            prefix, _, tail = key.partition(":")
            self._index.setdefault(prefix, {})[length] = tail
            self.stats.checkpoints += 1
            self._evict_to_fit()
            return True

    def _evict_to_fit(self, protect=None):
        while self._entries and self.bytes > self.capacity_bytes:
            key = next(iter(self._entries))
            if key == protect:
                # The caller is about to restore from this one. Freeing its blocks here would take
                # them out from under a restore already in flight -- which is how a host budget
                # smaller than a single checkpoint turned a hit into a KeyError halfway through.
                # Holding one checkpoint over the cap until the request has it is the cheaper miss.
                break
            checkpoint = self._entries.pop(key)
            self.stats.evictions += 1
            self._spill(key, checkpoint)
            self.store.release(checkpoint.block_keys())
            if key not in self._spilled:
                self._forget_index(key, checkpoint.length)

    def _forget_index(self, key, length):
        prefix, _, tail = key.partition(":")
        bucket = self._index.get(prefix)
        if bucket and bucket.get(length) == tail:
            bucket.pop(length, None)
            if not bucket:
                self._index.pop(prefix, None)

    def _spill(self, key, checkpoint):
        if self.spill_dir is None or self.spill_bytes <= 0:
            return
        if key in self._spilled:
            # Already on disk: this is an entry that was read back and has now aged out of the host
            # tier again. Rewriting the file would be wasted IO, and re-adding its size would count
            # the same bytes twice and shrink the storage tier every time a prefix is reused.
            self._spilled.move_to_end(key)
            return
        try:
            payload = self._materialize(checkpoint)
            path = self._path(key)
            torch.save(payload, path)
            size = path.stat().st_size
        except Exception as exc:  # noqa: BLE001 - a full disk must not fail a request
            self.stats.spill_errors += 1
            caps.announce_once(
                "prefix-spill-failed",
                f"could not write a prefix checkpoint to storage ({exc}); evicted checkpoints are "
                f"dropped instead. Reuse still works, over a shorter history.", logging.INFO)
            return
        self._spilled[key] = size
        self.spilled_bytes += size
        self.stats.spills += 1
        while self._spilled and self.spilled_bytes > self.spill_bytes:
            oldest, size = self._spilled.popitem(last=False)
            self.spilled_bytes -= size
            try:
                self._path(oldest).unlink()
            except OSError:
                pass

    def _materialize(self, checkpoint):
        """A checkpoint with its blocks inlined, for writing to storage.

        The host tier shares blocks between checkpoints; a file cannot, so this is where the
        sharing is paid for. It is the right trade at this tier: a spilled checkpoint is one that
        nothing has wanted recently, and storage is what it is being spilled to.
        """
        blocks = tuple((tuple(self.store.get(key) for key in keys),
                        tuple(self.store.get(key) for key in values))
                       for keys, values in checkpoint.blocks)
        return {"version": FORMAT_VERSION, "length": checkpoint.length, "kind": checkpoint.kind,
                "blocks": blocks, "residual": checkpoint.residual,
                "group_size": checkpoint.group_size,
                "residual_length": checkpoint.residual_length}

    def _load_spilled(self, key):
        if key not in self._spilled:
            return None
        try:
            payload = torch.load(self._path(key), map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 - a truncated or stale file is a miss, not a failure
            self._forget_spilled(key)
            return None
        if payload.get("version") != FORMAT_VERSION:
            self._forget_spilled(key)
            return None

        blocks = tuple((tuple(self.store.put(block, block.nbytes) for block in keys),
                        tuple(self.store.put(block, block.nbytes) for block in values))
                       for keys, values in payload["blocks"])
        residual = tuple(payload["residual"])
        checkpoint = Checkpoint(
            length=payload["length"], kind=payload["kind"], blocks=blocks, residual=residual,
            group_size=payload["group_size"], residual_length=payload["residual_length"],
            resident_bytes=sum(_tensor_bytes(k) + _tensor_bytes(v) for k, v in residual))
        self.store.acquire(checkpoint.block_keys())
        self._entries[key] = checkpoint
        self._spilled.move_to_end(key)
        self._evict_to_fit(protect=key)
        return checkpoint

    def _forget_spilled(self, key):
        size = self._spilled.pop(key, 0)
        self.spilled_bytes -= size
        try:
            self._path(key).unlink()
        except OSError:
            pass

    def _path(self, key):
        return self.spill_dir / f"{key}.pt"

    # -- lifecycle ---------------------------------------------------------------------------------

    def clear(self, drop_spilled=False):
        with self._lock:
            for checkpoint in self._entries.values():
                self.store.release(checkpoint.block_keys())
            self._entries.clear()
            if drop_spilled and self.spill_dir is not None:
                shutil.rmtree(self.spill_dir, ignore_errors=True)
                self._spilled.clear()
                self.spilled_bytes = 0
                try:
                    self.spill_dir.mkdir(parents=True, exist_ok=True)
                except OSError:
                    self.spill_dir = None

    def report(self):
        with self._lock:
            return self.stats.to_dict(self)


# ---- one request's use of it ---------------------------------------------------------------------

class PrefixSession:
    """The prefix cache as one generation sees it: restore up front, checkpoint as it runs.

    Lives for one request. Holds the running token list -- prompt plus what has been generated --
    because a checkpoint is keyed by the tokens it covers, and at the moment the cache reaches
    length N every one of those N tokens is already known: the token at position N-1 was sampled
    from the previous step and fed to this one.
    """

    def __init__(self, cache, tokens, config=None, layers=0, device=None):
        self.cache = cache
        self.config = config
        self.layers = layers
        self.device = device
        #: Grows as generation runs. The engine appends generated ids to it.
        self.tokens = list(tokens)
        self.prompt_length = len(self.tokens)
        self.match = None
        self.restored = 0
        self.prefilled = 0
        self.seconds_saved = 0.0
        self._memo = {}
        self._checkpointed = set()
        self._live = None

    # -- before generation --------------------------------------------------------------------------

    def begin(self, new_cache):
        """The cache this request should generate with, restored if a prefix was found.

        `new_cache` builds a fresh one, and is only called when there is nothing to restore.
        """
        if self.cache is None or not self.cache.enabled:
            return self._watch(new_cache())

        self.match = self.cache.lookup(self.tokens)
        if self.match is None:
            return self._watch(new_cache())
        try:
            restored = restore(self.match.checkpoint, self.cache.store, self.device, self.config,
                               self._memo)
        except Exception as exc:  # noqa: BLE001 - a bad checkpoint costs a prefill, not a request
            caps.announce_once(
                "prefix-restore-failed",
                f"could not restore a cached prefix ({exc}); this request prefills from scratch, "
                f"which is slower and produces the same answer.", logging.WARNING)
            self.match = None
            return self._watch(new_cache())
        self.restored = self.match.length
        return self._watch(restored)

    def _watch(self, cache):
        self._live = cache
        if cache is None or self.cache is None or not self.cache.enabled:
            return cache
        return checkpointing(cache, self._on_step, self.layers)

    # -- during generation ---------------------------------------------------------------------------

    def _on_step(self, cache):
        """A step finished. Checkpoint if this length is one worth being able to come back to.

        Two lengths are: a block boundary, and the end of the prompt. The second is not an
        optimisation -- it is the one that matters. A turn's prefill goes from nothing to the whole
        prompt in ONE pass, because on this engine a pass moves every weight through the device and
        chunking it to land on block boundaries would cost a full streaming pass per chunk. So the
        boundaries below the prompt length are never lengths the cache actually passes through, and
        checkpointing only at boundaries stores nothing a later turn can use. Measured before this
        was here: three turns of a growing conversation, one checkpoint, zero hits.
        """
        self._maybe_checkpoint(cache, cache.get_seq_length())

    def _maybe_checkpoint(self, cache, length, force=False):
        if self.cache is None or not self.cache.enabled or length <= 0:
            return False
        if length in self._checkpointed:
            return False
        interesting = force or length == self.prompt_length or not length % self.cache.block_size
        if not interesting:
            return False
        if length > len(self.tokens):
            # The token at the last position has not been sampled yet, so the key covering it
            # cannot be computed. Nothing is lost; the next interesting length is along shortly.
            return False
        self._checkpointed.add(length)
        try:
            checkpoint = capture(cache, self.cache.store, self._memo)
        except Exception:  # noqa: BLE001 - never let bookkeeping fail a generation
            log.debug("could not capture a prefix checkpoint at %d tokens", length, exc_info=True)
            return False
        if checkpoint is None or checkpoint.length != length:
            return False
        return self.cache.put(self.tokens, checkpoint)

    def observe_tokens(self, token_ids):
        """Generated tokens, as they are sampled. Keeps the running sequence complete."""
        self.tokens.extend(int(token) for token in token_ids)

    # -- after generation ------------------------------------------------------------------------------

    def finish(self, prefill_seconds=None):
        """Record what the reuse was worth, and checkpoint where the next turn will start.

        The end of a turn is the single most valuable length to hold: an agentic client's next
        prompt is this turn's whole conversation plus whatever it appends, so the prefix it shares
        is exactly what the cache holds right now.
        """
        # Recorded whether or not the cache is on, so a run with it off still reports honestly how
        # many tokens it prefilled rather than a column of zeros.
        self.prefilled = max(0, self.prompt_length - self.restored)
        if self.cache is None or not self.cache.enabled:
            return
        if self._live is not None:
            try:
                self._maybe_checkpoint(self._live, self._live.get_seq_length(), force=True)
            except Exception:  # noqa: BLE001 - bookkeeping never fails a completed generation
                log.debug("could not capture the end-of-turn prefix checkpoint", exc_info=True)
        with self.cache._lock:
            self.cache.stats.tokens_skipped += self.restored
            self.cache.stats.tokens_prefilled += self.prefilled
            if prefill_seconds is None:
                return
            if not self.restored:
                # A full prefill. This is what makes a baseline for the requests that skip one.
                self.cache.baseline.observe(self.prompt_length, prefill_seconds)
                return
            baseline = self.cache.baseline.estimate(self.prompt_length)
            if baseline is None:
                # Nothing comparable has run a full prefill on this machine yet, so there is no
                # honest number to give. Reported as unknown rather than guessed at.
                self.seconds_saved = None
                return
            self.seconds_saved = max(0.0, baseline - prefill_seconds)
            self.cache.stats.seconds_saved += self.seconds_saved

    def summary(self):
        return {
            "hit": self.match is not None,
            "tier": self.match.tier if self.match else None,
            "prompt_tokens": self.prompt_length,
            "tokens_reused": self.restored,
            "tokens_prefilled": self.prefilled,
            "seconds_saved": (None if self.seconds_saved is None
                              else round(self.seconds_saved, 3)),
        }


PREFIX_AUTO = "auto"
PREFIX_ON = "on"
PREFIX_OFF = "off"
PREFIX_CHOICES = (PREFIX_AUTO, PREFIX_ON, PREFIX_OFF)


def resolve(setting, weight_bytes=None, device_bytes=None, headroom=0.15):
    """Whether reusing a prefix is worth what it costs on THIS machine. Returns (enabled, why).

    This is regime-dependent, and in the opposite direction to the obvious guess.

    A prefill is ONE streaming pass. Every weight moves through the device exactly once whether the
    prompt is forty tokens or four thousand, so on a machine where the model does not fit, skipping
    tokens does not skip the thing that costs -- and the checkpoint still has to be copied out to
    the host. Measured on an RTX 3090, TinyLlama, weight cache squeezed to 1GB: three turns of a
    1800-token conversation were 8.6% SLOWER in prefill with reuse on, having reused 3741 of 5634
    prompt tokens. The reuse was real; there was simply nothing under it to save.

    Where the weights are resident the pass is free and prefill is compute -- attention over the
    prompt, which grows with its square. There, skipping the prompt skips real work: the same
    conversation with the model resident was 14-18% faster per reused turn.

    So "auto" follows the same measured recommendation speculative decoding does, for the same
    reason: a feature that is net-negative in one regime must not switch itself on there silently.
    """
    if setting == PREFIX_ON:
        return True, "forced on"
    if setting == PREFIX_OFF:
        return False, "forced off"
    if setting != PREFIX_AUTO:
        raise ValueError(f"prefix_cache must be one of {', '.join(PREFIX_CHOICES)}, not {setting!r}")

    if not weight_bytes or not device_bytes:
        return False, ("the model's size against the device budget could not be read, and reuse "
                       "costs more than it saves wherever a prefill is dominated by streaming "
                       "weights rather than by compute")
    fits = weight_bytes * (1.0 + headroom) <= device_bytes
    if fits:
        return True, (f"the weights fit resident ({weight_bytes / 1024 ** 3:.1f}GB against a "
                      f"{device_bytes / 1024 ** 3:.1f}GB budget), so a prefill is compute and "
                      f"skipping the prompt skips real work")
    return False, (f"the weights do not fit resident ({weight_bytes / 1024 ** 3:.1f}GB against a "
                   f"{device_bytes / 1024 ** 3:.1f}GB budget), so a prefill is one streaming pass "
                   f"whose cost does not depend on the prompt length -- reuse would pay for "
                   f"checkpoints and save nothing")


def build(profile=None, seed=b"", enabled=True, spill_dir=None, block_size=None,
          capacity_bytes=None, spill_bytes=None):
    """A prefix cache sized from the machine, with every knob overridable for debugging."""
    def knob(name, fallback):
        if profile is not None:
            derivation = profile.derived.get(name)
            if derivation is not None:
                return int(derivation.value)
        return fallback

    if spill_dir is None:
        from ..hw.profile import user_cache_dir

        spill_dir = user_cache_dir() / "prefix"
    return PrefixCache(
        block_size=block_size if block_size is not None else knob("prefix_block_tokens", 256),
        capacity_bytes=(capacity_bytes if capacity_bytes is not None
                        else knob("prefix_cache_bytes", 0)),
        spill_bytes=spill_bytes if spill_bytes is not None else knob("prefix_spill_bytes", 0),
        spill_dir=spill_dir, seed=seed, enabled=enabled)
