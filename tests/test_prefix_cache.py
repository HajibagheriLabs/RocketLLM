"""Tests for reusing the KV cache of a prefix a client has already sent.

Everything here runs on CPU in a few seconds. The end-to-end tests use a two-layer,
64-token-vocabulary toy model, because none of what is being checked depends on the model being
large -- what it depends on is the cache layout, and that is the same at any size.

The test this file exists for is the int4 boundary. The quantized cache splits its history into
int4 blocks plus an fp16 residual window of the most recent tokens, and which side a token falls on
is a function of the length alone. Restore a cache whose split does not match what a real prefill
would have produced and the tokens near the boundary are quantized where they should be exact:
attention still runs, nothing raises, and the answer is slightly wrong from there on. So the split
is checked against a from-scratch prefill directly, structurally, rather than being inferred from
output that happened to look reasonable.

The other one is the gate the whole feature has to pass: for the same conversation, greedy output
with the prefix cache on has to match greedy output with it off, token for token. Reuse is a cost
optimisation; if it changes an answer it is a bug.
"""
import unittest

import torch

from rocketllm.quant.kv_cache import KVCacheConfig, QuantizedKVCache
from rocketllm.server import prefix_cache as pc

VOCAB = 64
LAYERS = 2
GROUP = 64
RESIDUAL = 128


def kv_config(group=GROUP, residual=RESIDUAL):
    return KVCacheConfig(group_size=group, residual_length=residual, compute_dtype=torch.float32)


def tiny_model(seed=0, layers=LAYERS):
    """A real transformers model, small enough to be free and real enough to exercise the cache."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=4096)
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def prefilled(length, layers=LAYERS, seed=0, config=None):
    """A quantized cache holding `length` tokens, filled in one pass like a real prefill."""
    torch.manual_seed(seed)
    cache = QuantizedKVCache(config or kv_config())
    for layer in range(layers):
        keys = torch.randn(1, 2, length, 8)
        values = torch.randn(1, 2, length, 8)
        cache.update(keys, values, layer)
    return cache


def split_of(cache):
    """(quantized tokens, residual tokens) per layer -- the thing that must not move."""
    return [(sum(block.length for block in cache._blocks[layer][0]),
             cache.key_cache[layer].shape[-2])
            for layer in range(len(cache.key_cache))]


def full_history(cache):
    """Every layer's K and V as attention would see them, blocks expanded."""
    out = []
    for layer in range(len(cache.key_cache)):
        keys, values = cache._blocks[layer]
        out.append((
            torch.cat([block.dequantize() for block in keys] + [cache.key_cache[layer]], dim=-2),
            torch.cat([block.dequantize() for block in values] + [cache.value_cache[layer]],
                      dim=-2)))
    return out


# ---- hashing -----------------------------------------------------------------------------------

class TestBlockHashing(unittest.TestCase):

    def test_a_shared_prefix_shares_its_chain(self):
        left = list(range(1000))
        right = list(range(600)) + [999] * 400
        a = pc.chain_hashes(left, 256)
        b = pc.chain_hashes(right, 256)
        # Two whole blocks are identical; the third contains the divergence.
        self.assertEqual(a[:3], b[:3])
        self.assertNotEqual(a[3], b[3])

    def test_a_changed_token_changes_every_hash_after_it(self):
        """Chained, not per-block: a match at depth k has to mean the first k blocks are identical,
        not that block k happens to be."""
        base = list(range(1000))
        changed = list(base)
        changed[10] = 999
        a, b = pc.chain_hashes(base, 256), pc.chain_hashes(changed, 256)
        self.assertEqual(a[0], b[0])                     # the empty prefix
        self.assertTrue(all(x != y for x, y in zip(a[1:], b[1:])))

    def test_the_empty_prefix_is_index_zero(self):
        self.assertEqual(len(pc.chain_hashes(list(range(700)), 256)), 3)  # root + 2 whole blocks

    def test_a_partial_block_does_not_produce_a_hash(self):
        self.assertEqual(pc.chain_hashes(list(range(255)), 256),
                         pc.chain_hashes([], 256))

    def test_different_seeds_never_agree(self):
        tokens = list(range(512))
        self.assertNotEqual(pc.chain_hashes(tokens, 256, b"model-a"),
                            pc.chain_hashes(tokens, 256, b"model-b"))

    def test_the_namespace_seed_separates_models(self):
        """Two models produce entirely different KV for the same tokens. Sharing a spill directory
        must not let one restore the other's cache."""
        self.assertNotEqual(pc.namespace_seed("llama", "bfloat16", "int4"),
                            pc.namespace_seed("qwen", "bfloat16", "int4"))
        self.assertNotEqual(pc.namespace_seed("llama", "bfloat16", "int4"),
                            pc.namespace_seed("llama", "float16", "int4"))

    def test_token_ids_cannot_run_together(self):
        """Fixed-width encoding: [1, 2] and [258] must not hash alike."""
        self.assertNotEqual(pc.tail_hash([1, 2]), pc.tail_hash([258]))


# ---- the int4 boundary -------------------------------------------------------------------------

class TestTheQuantizedBoundary(unittest.TestCase):
    """The failure this design exists to prevent, checked structurally."""

    #: Lengths chosen around the two numbers that decide the split: the residual window and the
    #: group size. Below the window nothing is quantized; each group past it moves one more group.
    LENGTHS = (1, 64, 127, 128, 129, 191, 192, 193, 255, 256, 300, 512, 700, 1024)

    def test_a_restored_cache_has_the_same_split_as_a_fresh_prefill(self):
        for length in self.LENGTHS:
            with self.subTest(length=length):
                store = pc._BlockStore()
                checkpoint = pc.capture(prefilled(length), store)
                restored = pc.restore(checkpoint, store, "cpu", kv_config())
                self.assertEqual(split_of(restored), split_of(prefilled(length)))

    def test_the_split_is_the_one_the_arithmetic_predicts(self):
        """quantized(n) = max(0, ((n - residual) // group) * group). Stated here so that a change
        to the flush rule breaks this test rather than the answers."""
        for length in self.LENGTHS:
            with self.subTest(length=length):
                expected = max(0, ((length - RESIDUAL) // GROUP) * GROUP)
                for quantized, residual in split_of(prefilled(length)):
                    self.assertEqual(quantized, expected)
                    self.assertEqual(quantized + residual, length)

    def test_a_restored_cache_holds_the_same_numbers(self):
        for length in self.LENGTHS:
            with self.subTest(length=length):
                store = pc._BlockStore()
                fresh = prefilled(length)
                restored = pc.restore(pc.capture(fresh, store), store, "cpu", kv_config())
                for (fresh_k, fresh_v), (got_k, got_v) in zip(full_history(fresh),
                                                              full_history(restored)):
                    self.assertTrue(torch.equal(fresh_k, got_k))
                    self.assertTrue(torch.equal(fresh_v, got_v))

    def test_the_residual_window_holds_the_same_tokens_exactly(self):
        """The window is the part that is NOT quantized. If a restore puts the boundary a group out,
        tokens that should be exact come back rounded and nothing anywhere reports it."""
        for length in self.LENGTHS:
            with self.subTest(length=length):
                store = pc._BlockStore()
                fresh = prefilled(length)
                restored = pc.restore(pc.capture(fresh, store), store, "cpu", kv_config())
                for layer in range(LAYERS):
                    self.assertTrue(torch.equal(fresh.key_cache[layer],
                                                restored.key_cache[layer]))
                    self.assertTrue(torch.equal(fresh.value_cache[layer],
                                                restored.value_cache[layer]))

    def test_restoring_under_a_different_layout_is_refused(self):
        """group_size and residual_length are what decide the split. Restoring under different ones
        would put the boundary somewhere the running cache never would, and nothing downstream
        would notice -- so it raises rather than proceeding."""
        store = pc._BlockStore()
        checkpoint = pc.capture(prefilled(512), store)
        with self.assertRaises(ValueError) as caught:
            pc.restore(checkpoint, store, "cpu", kv_config(residual=256))
        self.assertIn("boundary", str(caught.exception))

    def test_continuing_from_a_restored_cache_does_not_rewrite_the_checkpoint(self):
        """Blocks are shared with the checkpoint that holds them. Appending has to extend a copy of
        the list, or a later restore would find history that was written after it was promised."""
        store = pc._BlockStore()
        checkpoint = pc.capture(prefilled(300), store)
        before = [(len(keys), len(values)) for keys, values in checkpoint.blocks]

        restored = pc.restore(checkpoint, store, "cpu", kv_config())
        for layer in range(LAYERS):
            restored.update(torch.randn(1, 2, 400, 8), torch.randn(1, 2, 400, 8), layer)

        self.assertEqual([(len(k), len(v)) for k, v in checkpoint.blocks], before)
        self.assertEqual(checkpoint.length, 300)
        self.assertEqual(restored.get_seq_length(), 700)

    def test_a_checkpoint_is_only_ever_restored_at_its_own_length(self):
        """The reason there is no "truncate a longer cache" path anywhere: truncating would have to
        dequantize tokens back into the window, and a dequantized token is not the token a real
        prefill would have had there."""
        cache = pc.PrefixCache(block_size=64, capacity_bytes=1 << 30, spill_dir=None)
        tokens = list(range(500))
        store = cache.store
        checkpoint = pc.capture(prefilled(300), store)
        checkpoint.length = 300
        cache.put(tokens, checkpoint)

        match = cache.lookup(tokens)
        self.assertIsNotNone(match)
        self.assertEqual(match.length, checkpoint.length)


class TestTheFullPrecisionLayout(unittest.TestCase):
    """The other cache the server may be running. Simpler, and it still has to round-trip."""

    def test_capture_and_restore_are_exact(self):
        from transformers.cache_utils import DynamicCache

        torch.manual_seed(0)
        cache = DynamicCache()
        for layer in range(LAYERS):
            cache.update(torch.randn(1, 2, 300, 8), torch.randn(1, 2, 300, 8), layer)

        restored = pc.restore(pc.capture(cache, pc._BlockStore()), pc._BlockStore(), "cpu")
        self.assertEqual(restored.get_seq_length(), 300)
        for layer in range(LAYERS):
            self.assertTrue(torch.equal(cache.key_cache[layer], restored.key_cache[layer]))
            self.assertTrue(torch.equal(cache.value_cache[layer], restored.value_cache[layer]))


# ---- the cache itself --------------------------------------------------------------------------

def checkpoint_at(length, store, config=None):
    return pc.capture(prefilled(length, config=config), store)


def entry_size(length=300):
    """What one checkpoint costs the host tier: its residual window plus its quantized blocks.

    Measured rather than guessed, because a capacity that forgets the blocks is a capacity no
    single entry can fit inside, and every test built on it would evict everything and pass for
    the wrong reason.
    """
    store = pc._BlockStore()
    checkpoint = checkpoint_at(length, store)
    return store.bytes + checkpoint.resident_bytes


class TestLookup(unittest.TestCase):

    def cache(self, **kwargs):
        kwargs.setdefault("block_size", 64)
        kwargs.setdefault("capacity_bytes", 1 << 30)
        kwargs.setdefault("spill_dir", None)
        return pc.PrefixCache(**kwargs)

    def test_a_miss_on_an_empty_cache(self):
        self.assertIsNone(self.cache().lookup(list(range(500))))

    def test_an_exact_prefix_is_found(self):
        cache = self.cache()
        tokens = list(range(500))
        cache.put(tokens, checkpoint_at(300, cache.store))
        match = cache.lookup(tokens + list(range(1000, 1100)))
        self.assertIsNotNone(match)
        self.assertEqual(match.length, 300)

    def test_a_length_that_is_not_a_block_boundary_is_still_indexable(self):
        """Which is the case that matters: one prefill pass lands wherever the prompt ends, and
        chunking it to land on a boundary would cost a whole streaming pass per block."""
        cache = self.cache(block_size=256)
        tokens = list(range(1000))
        cache.put(tokens, checkpoint_at(700, cache.store))
        match = cache.lookup(tokens)
        self.assertEqual(match.length, 700)
        self.assertNotEqual(700 % 256, 0)

    def test_the_longest_of_several_stored_prefixes_wins(self):
        cache = self.cache()
        tokens = list(range(1000))
        for length in (200, 500, 800):
            cache.put(tokens, checkpoint_at(length, cache.store))
        self.assertEqual(cache.lookup(tokens).length, 800)

    def test_a_shorter_prefix_is_used_when_the_longest_no_longer_matches(self):
        """A client that edits or branches its history. The partial match is the point of hashing
        in blocks at all."""
        cache = self.cache()
        tokens = list(range(1000))
        cache.put(tokens, checkpoint_at(200, cache.store))
        cache.put(tokens, checkpoint_at(800, cache.store))

        branched = tokens[:300] + [7] * 700
        match = cache.lookup(branched)
        self.assertIsNotNone(match)
        self.assertEqual(match.length, 200)

    def test_a_prefix_that_is_not_a_prefix_is_not_a_hit(self):
        cache = self.cache()
        cache.put(list(range(1000)), checkpoint_at(300, cache.store))
        self.assertIsNone(cache.lookup([9] * 1000))

    def test_a_checkpoint_longer_than_the_request_is_not_a_hit(self):
        cache = self.cache()
        cache.put(list(range(1000)), checkpoint_at(800, cache.store))
        self.assertIsNone(cache.lookup(list(range(500))))

    def test_a_disabled_cache_never_hits(self):
        cache = self.cache(capacity_bytes=0)
        self.assertFalse(cache.enabled)
        tokens = list(range(500))
        self.assertFalse(cache.put(tokens, checkpoint_at(300, cache.store)))
        self.assertIsNone(cache.lookup(tokens))


class TestEviction(unittest.TestCase):
    """LRU, and deliberately so. See the module docstring in prefix_cache.py: the layer cache
    refuses LRU because decoder layers are read cyclically, and on a cyclic scan LRU evicts exactly
    what is needed next. Prefixes are recency-driven, which is what LRU is actually for."""

    def cache(self, capacity):
        return pc.PrefixCache(block_size=64, capacity_bytes=capacity, spill_dir=None)

    def test_the_oldest_goes_first(self):
        cache = self.cache(entry_size() * 3)
        for start in (0, 1000, 2000, 3000):
            cache.put(list(range(start, start + 500)), checkpoint_at(300, cache.store))
        self.assertIsNone(cache.lookup(list(range(0, 500))))
        self.assertIsNotNone(cache.lookup(list(range(3000, 3500))))

    def test_using_an_entry_keeps_it(self):
        cache = self.cache(entry_size() * 3)
        oldest = list(range(0, 500))
        cache.put(oldest, checkpoint_at(300, cache.store))
        for start in (1000, 2000):
            cache.put(list(range(start, start + 500)), checkpoint_at(300, cache.store))

        self.assertIsNotNone(cache.lookup(oldest))          # touch it
        cache.put(list(range(3000, 3500)), checkpoint_at(300, cache.store))
        self.assertIsNotNone(cache.lookup(oldest), "a recently used entry was evicted first")

    def test_shared_blocks_are_counted_once(self):
        """Checkpoints of one conversation hold overlapping prefixes of one append-only list.
        Charging each of them the whole history would evict almost everything."""
        cache = self.cache(1 << 30)
        live = prefilled(700)
        memo = {}
        early = pc.capture(live, cache.store, memo)
        after_one = cache.store.bytes
        again = pc.capture(live, cache.store, memo)
        self.assertEqual(cache.store.bytes, after_one, "the same blocks were copied twice")
        self.assertEqual(early.blocks, again.blocks)

    def test_blocks_are_released_when_the_last_holder_is_evicted(self):
        cache = self.cache(1 << 30)
        cache.put(list(range(500)), checkpoint_at(300, cache.store))
        self.assertGreater(cache.store.bytes, 0)
        cache.clear()
        self.assertEqual(cache.store.bytes, 0)
        self.assertEqual(cache.bytes, 0)


class TestSpilling(unittest.TestCase):

    def test_an_evicted_checkpoint_survives_on_storage(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            cache = pc.PrefixCache(block_size=64, capacity_bytes=entry_size(),
                                   spill_bytes=1 << 30, spill_dir=directory)
            first = list(range(500))
            cache.put(first, checkpoint_at(300, cache.store))
            for start in (1000, 2000, 3000):
                cache.put(list(range(start, start + 500)), checkpoint_at(300, cache.store))

            self.assertGreater(cache.stats.spills, 0, "nothing was written to storage")
            match = cache.lookup(first)
            self.assertIsNotNone(match, "a spilled checkpoint could not be read back")
            self.assertEqual(match.tier, "storage")
            self.assertEqual(match.length, 300)

    def test_a_reloaded_checkpoint_still_restores_exactly(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            cache = pc.PrefixCache(block_size=64, capacity_bytes=1, spill_bytes=1 << 30,
                                   spill_dir=directory)
            tokens = list(range(500))
            fresh = prefilled(300)
            cache.put(tokens, pc.capture(fresh, cache.store))

            match = cache.lookup(tokens)
            self.assertEqual(match.tier, "storage")
            restored = pc.restore(match.checkpoint, cache.store, "cpu", kv_config())
            self.assertEqual(split_of(restored), split_of(fresh))
            for (a_k, a_v), (b_k, b_v) in zip(full_history(fresh), full_history(restored)):
                self.assertTrue(torch.equal(a_k, b_k))
                self.assertTrue(torch.equal(a_v, b_v))

    def test_no_spill_directory_is_not_an_error(self):
        cache = pc.PrefixCache(block_size=64, capacity_bytes=entry_size(), spill_dir=None)
        for start in (0, 1000, 2000):
            cache.put(list(range(start, start + 500)), checkpoint_at(300, cache.store))
        self.assertEqual(cache.stats.spills, 0)
        self.assertIsNotNone(cache.lookup(list(range(2000, 2500))))


# ---- end to end --------------------------------------------------------------------------------

class Conversation:
    """Drives turns through a PrefixSession the way the server does."""

    def __init__(self, cache, kind="int4", layers=LAYERS, model=None):
        self.cache = cache
        self.kind = kind
        self.layers = layers
        self.model = model or tiny_model()
        self.config = kv_config() if kind == "int4" else None
        self.sessions = []

    def _fresh(self):
        from transformers.cache_utils import DynamicCache

        return QuantizedKVCache(self.config) if self.kind == "int4" else DynamicCache()

    def turn(self, tokens, new_tokens=8):
        session = pc.PrefixSession(self.cache, tokens, config=self.config, layers=self.layers,
                                   device="cpu")
        cache = session.begin(self._fresh)
        ids = torch.tensor([tokens])

        class Tap:
            """The streamer the server uses, reduced to the one thing the session needs of it."""

            def __init__(self):
                self.prompt_seen = False

            def put(self, value):
                if not self.prompt_seen:
                    self.prompt_seen = True
                    return
                session.observe_tokens(value.reshape(-1).tolist())

            def end(self):
                pass

        out = self.model.generate(ids, max_new_tokens=new_tokens, do_sample=False, pad_token_id=0,
                                  past_key_values=cache, attention_mask=torch.ones_like(ids),
                                  streamer=Tap())
        session.finish(prefill_seconds=0.01)
        self.sessions.append(session)
        return out[0].tolist()


def run_conversation(cache, turns=3, prompt=400, appended=120, kind="int4", model=None, seed=1):
    """A client that resends everything every turn, which is what an agentic one does."""
    torch.manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (prompt,)).tolist()
    convo = Conversation(cache, kind=kind, model=model)
    outputs = []
    for _ in range(turns):
        out = convo.turn(tokens)
        outputs.append(out)
        tokens = out + torch.randint(0, VOCAB, (appended,)).tolist()
    return outputs, convo


class TestTheCorrectnessGate(unittest.TestCase):
    """THE test. Reuse is a cost optimisation; if it changes a token it is a bug."""

    def gate(self, kind):
        model = tiny_model()
        on = pc.PrefixCache(block_size=128, capacity_bytes=1 << 30, spill_dir=None, seed=b"gate")
        off = pc.PrefixCache(enabled=False)
        with_cache, session = run_conversation(on, kind=kind, model=model)
        without, _ = run_conversation(off, kind=kind, model=model)
        return with_cache, without, session

    def test_greedy_output_is_identical_with_the_prefix_cache_and_without_it_int4(self):
        with_cache, without, convo = self.gate("int4")
        self.assertEqual(with_cache, without)
        reused = [session.restored for session in convo.sessions]
        self.assertGreater(sum(reused), 0, "nothing was reused, so this proved nothing")

    def test_greedy_output_is_identical_with_the_prefix_cache_and_without_it_fp16(self):
        with_cache, without, convo = self.gate("fp16")
        self.assertEqual(with_cache, without)
        self.assertGreater(sum(session.restored for session in convo.sessions), 0)

    def test_it_holds_when_the_conversation_branches(self):
        """The partial-match path: the client rewrites the tail, so the deepest checkpoint is no
        longer a prefix and a shorter one has to be used instead."""
        model = tiny_model()
        cache = pc.PrefixCache(block_size=128, capacity_bytes=1 << 30, spill_dir=None)
        torch.manual_seed(3)
        base = torch.randint(0, VOCAB, (500,)).tolist()

        first = Conversation(cache, model=model)
        first.turn(base)

        branched = base[:300] + torch.randint(0, VOCAB, (200,)).tolist()
        with_cache = Conversation(cache, model=model).turn(branched)
        without = Conversation(pc.PrefixCache(enabled=False), model=model).turn(branched)
        self.assertEqual(with_cache, without)


class TestReuse(unittest.TestCase):
    """That the thing actually saves the work it claims to."""

    def test_each_turn_after_the_first_reuses_nearly_all_of_the_previous_one(self):
        cache = pc.PrefixCache(block_size=128, capacity_bytes=1 << 30, spill_dir=None)
        _, convo = run_conversation(cache, turns=3, prompt=400, appended=120)
        reused = [session.restored for session in convo.sessions]
        prompts = [session.prompt_length for session in convo.sessions]

        self.assertEqual(reused[0], 0, "there was nothing to reuse on the first turn")
        for turn in (1, 2):
            with self.subTest(turn=turn):
                # Everything but the tokens the client appended, less the one token whose KV was
                # never computed: the last token a turn emits is sampled but never fed back in.
                self.assertGreaterEqual(reused[turn], prompts[turn] - 121)

    def test_the_stats_add_up(self):
        cache = pc.PrefixCache(block_size=128, capacity_bytes=1 << 30, spill_dir=None)
        _, convo = run_conversation(cache, turns=3)
        report = cache.report()
        self.assertEqual(report["lookups"], 3)
        self.assertEqual(report["hits"], 2)
        self.assertEqual(report["tokens_skipped"],
                         sum(session.restored for session in convo.sessions))
        self.assertGreater(report["tokens_reused_fraction"], 0.5)
        self.assertIn("seconds_saved_vs_measured_baseline", report)


class TestWhenItIsWorthHaving(unittest.TestCase):
    """The regime question, and it runs the opposite way to the obvious guess.

    A prefill is ONE streaming pass: every weight crosses the link once whether the prompt is forty
    tokens or four thousand. So on a machine where the model does not fit, skipping prompt tokens
    does not skip what costs -- measured at 8.6% SLOWER over three turns with 3741 of 5634 tokens
    genuinely reused. Where the weights are resident the pass is free and prefill is compute, and
    the same conversation ran 14-18% faster per reused turn.
    """

    def test_resident_weights_recommend_it(self):
        enabled, reason = pc.resolve("auto", weight_bytes=2 << 30, device_bytes=21 << 30)
        self.assertTrue(enabled)
        self.assertIn("compute", reason)

    def test_weights_that_do_not_fit_recommend_against_it(self):
        enabled, reason = pc.resolve("auto", weight_bytes=40 << 30, device_bytes=8 << 30)
        self.assertFalse(enabled)
        self.assertIn("streaming pass", reason)

    def test_unknown_sizes_get_no_for_an_answer(self):
        """Never switch on a feature that is net-negative in one regime without knowing which one
        this is -- the same rule speculative decoding follows."""
        self.assertFalse(pc.resolve("auto")[0])
        self.assertFalse(pc.resolve("auto", weight_bytes=1 << 30)[0])

    def test_explicit_settings_override_the_recommendation_both_ways(self):
        self.assertTrue(pc.resolve("on", weight_bytes=40 << 30, device_bytes=8 << 30)[0])
        self.assertFalse(pc.resolve("off", weight_bytes=2 << 30, device_bytes=21 << 30)[0])

    def test_an_unknown_setting_is_refused(self):
        with self.assertRaises(ValueError):
            pc.resolve("sometimes")


class TestTimeSavedIsNotInvented(unittest.TestCase):
    """What a saving is allowed to be measured against.

    The first version of this multiplied the request's own per-token prefill cost by the tokens it
    skipped. Prefill is parallel across tokens, so that rate collapses with length: on a 1.1B model
    with the weights resident, 38 tokens took 80ms (2.1ms each) while a full 1878-token prefill took
    100ms (0.05ms each). The extrapolation claimed 7.92s saved over three turns where the measured
    saving was 0.04s.
    """

    def test_nothing_is_claimed_until_a_comparable_prefill_has_been_measured(self):
        baseline = pc._PrefillBaseline()
        self.assertIsNone(baseline.estimate(1800))

    def test_a_prefill_of_a_different_size_is_not_a_baseline(self):
        baseline = pc._PrefillBaseline()
        baseline.observe(40, 0.08)
        self.assertIsNone(baseline.estimate(1800))

    def test_a_comparable_prefill_is(self):
        baseline = pc._PrefillBaseline()
        baseline.observe(1800, 0.10)
        baseline.observe(1900, 0.12)
        self.assertAlmostEqual(baseline.estimate(1850), 0.11, places=6)

    def test_a_session_with_no_baseline_reports_unknown_rather_than_a_number(self):
        cache = pc.PrefixCache(block_size=128, capacity_bytes=1 << 30, spill_dir=None)
        session = pc.PrefixSession(cache, list(range(500)), layers=LAYERS, device="cpu")
        session.restored = 400
        session.finish(prefill_seconds=0.01)
        self.assertIsNone(session.summary()["seconds_saved"])

    def test_the_saving_is_the_difference_against_that_baseline(self):
        cache = pc.PrefixCache(block_size=128, capacity_bytes=1 << 30, spill_dir=None)
        cache.baseline.observe(500, 0.10)

        session = pc.PrefixSession(cache, list(range(500)), layers=LAYERS, device="cpu")
        session.restored = 460
        session.finish(prefill_seconds=0.04)
        self.assertAlmostEqual(session.summary()["seconds_saved"], 0.06, places=6)

    def test_a_disabled_cache_reuses_nothing_and_still_answers(self):
        cache = pc.PrefixCache(enabled=False)
        outputs, convo = run_conversation(cache, turns=2)
        self.assertEqual([session.restored for session in convo.sessions], [0, 0])
        self.assertTrue(all(outputs))

    def test_an_unrelated_conversation_does_not_hit(self):
        cache = pc.PrefixCache(block_size=128, capacity_bytes=1 << 30, spill_dir=None)
        run_conversation(cache, turns=2, seed=1)
        _, other = run_conversation(cache, turns=1, seed=99)
        self.assertEqual(other.sessions[0].restored, 0)


if __name__ == "__main__":
    unittest.main()
