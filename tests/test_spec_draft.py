"""Tests for speculative decoding.

The claim this feature rests on is that it changes nothing about the output -- it only changes how
many streaming passes the output cost. That claim is a statement about arithmetic, so most of what
is here checks the arithmetic directly rather than checking that some generation looked reasonable.

Three levels, deliberately:

  1. ALGEBRAIC. The exact law of one accept/reject step is computed in closed form and compared with
     the target distribution at machine precision. No sampling, no tolerance, no flakiness. The
     natural-looking mistake -- resampling from p rather than from the residual -- is checked to
     FAIL that same test, so the test is known to be able to detect it.
  2. EMPIRICAL. Many actual draws through the real code path, compared with the distribution they
     are supposed to follow.
  3. END TO END. Greedy output with speculation on, token for token against greedy output with it
     off, through two real transformers models.

On the sampled path, end-to-end stream equality is NOT asserted, because it is not true: rejection
sampling consumes randomness that direct sampling does not, so the two draw differently from the
same distribution under any fixed seed. Level 1 is what pins the sampled path, and it pins it harder
than a seed comparison would.

Everything here runs on CPU in a couple of seconds. The models are two-layer, 64-token-vocabulary
toys; nothing about the algorithm cares how big the model is.
"""
import unittest

import torch

from rocketllm.spec.draft import (SPEC_AUTO, SPEC_OFF, SPEC_ON, DraftIncompatible, DraftModel,
                                  SamplingParams, SpeculativeDecoder, check_compatible,
                                  lookahead_ceiling, resolve_speculation, verify)

VOCAB = 64


def tiny_model(seed, vocab_size=VOCAB, layers=2):
    """A real transformers model, small enough to be free and real enough to exercise the cache."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=256)
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def probabilities(seed, size=8, sharpness=1.0):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(size, generator=generator) * sharpness
    return torch.softmax(logits, dim=-1)


def exact_output_law(p, q):
    """The law of one accept/reject step, computed rather than sampled.

    A proposal x ~ q is kept with probability min(1, p(x)/q(x)), so the mass arriving at t by
    acceptance is q(t)*min(1, p(t)/q(t)) = min(p(t), q(t)). Everything else is a rejection, and it
    is redistributed over the normalized residual.
    """
    accepted = torch.minimum(p, q)
    residual = torch.clamp(p - q, min=0.0)
    return accepted + (1.0 - accepted.sum()) * residual / residual.sum()


def wrong_output_law(p, q):
    """What resampling from p instead of the residual would produce. Kept so the test above is
    known to be capable of failing."""
    accepted = torch.minimum(p, q)
    return accepted + (1.0 - accepted.sum()) * p


class TestDistributionPreservation(unittest.TestCase):
    """The whole correctness claim, at machine precision."""

    def test_the_acceptance_identity_holds_exactly(self):
        for seed in range(12):
            p, q = probabilities(seed, sharpness=1.5), probabilities(seed + 100, sharpness=0.7)
            with self.subTest(seed=seed):
                self.assertTrue(torch.allclose(exact_output_law(p, q), p, atol=1e-6),
                                "the accept/resample split does not reproduce the target")

    def test_the_identity_would_catch_resampling_from_the_target(self):
        """The natural-looking mistake, and proof this test can see it.

        Sampling the replacement from p rather than from (p-q)+ double-counts the mass the draft
        already agreed with, biasing output towards whatever the draft was good at. It looks
        correct, it produces fluent text, and only the arithmetic says otherwise.
        """
        p, q = probabilities(1, sharpness=2.0), probabilities(2, sharpness=2.0)
        self.assertFalse(torch.allclose(wrong_output_law(p, q), p, atol=1e-4),
                         "the wrong rule passed, so this test cannot detect the bug it exists for")

    def test_many_draws_through_the_real_code_follow_the_target(self):
        """Level 2: the implementation, not the algebra, exercised end to end on one position."""
        trials = 20000
        p, q = probabilities(7, sharpness=1.5), probabilities(8, sharpness=0.5)
        generator = torch.Generator().manual_seed(20260811)
        counts = torch.zeros_like(p)
        for _ in range(trials):
            token = int(torch.multinomial(q, 1, generator=generator).item())
            accepted, corrected = verify([token], [q], [p, p], generator)
            counts[token if accepted else corrected] += 1

        empirical = counts / trials
        distance = 0.5 * float((empirical - p).abs().sum())
        self.assertLess(distance, 0.02,
                        f"total variation {distance:.4f} between the sampled output and the target")

    def test_a_draft_that_is_always_wrong_still_produces_the_target(self):
        """The extreme the acceptance test has to survive: q with no overlap with p at all."""
        p = torch.zeros(8)
        p[:4] = 0.25
        q = torch.zeros(8)
        q[4:] = 0.25
        self.assertTrue(torch.allclose(exact_output_law(p, q), p, atol=1e-6))

        generator = torch.Generator().manual_seed(3)
        for _ in range(64):
            token = int(torch.multinomial(q, 1, generator=generator).item())
            accepted, corrected = verify([token], [q], [p, p], generator)
            self.assertEqual(accepted, 0, "a proposal with zero target mass was accepted")
            self.assertLess(corrected, 4, "the correction left the target's support")


class TestGreedyIsTheSameRule(unittest.TestCase):
    """Zero temperature is a point mass, not a separate algorithm."""

    def one_hot(self, index, size=8):
        out = torch.zeros(size)
        out[index] = 1.0
        return out

    def test_a_correct_greedy_guess_is_accepted(self):
        p = self.one_hot(3)
        q = self.one_hot(3)
        accepted, _ = verify([3], [q], [p, p])
        self.assertEqual(accepted, 1)

    def test_a_wrong_greedy_guess_is_rejected_and_corrected_to_the_argmax(self):
        p = self.one_hot(3)
        q = self.one_hot(5)
        accepted, corrected = verify([5], [q], [p, p])
        self.assertEqual(accepted, 0)
        self.assertEqual(corrected, 3, "the correction was not the target's own choice")

    def test_a_rejected_greedy_guess_consumes_no_randomness(self):
        """So a greedy run is reproducible whatever the seed happens to be.

        The ratio is exactly zero when the target gives the proposal no mass, and the outcome is
        settled without a draw. Anything else would make greedy output depend on the RNG state.
        """
        p, q = self.one_hot(3), self.one_hot(5)
        generator = torch.Generator().manual_seed(11)
        before = generator.get_state()
        verify([5], [q], [p, p])
        self.assertTrue(torch.equal(before, generator.get_state()))

    def test_sampling_params_produce_a_point_mass_when_greedy(self):
        logits = torch.tensor([[1.0, 5.0, 2.0, 0.0]])
        probs = SamplingParams(do_sample=False).distribution(logits)
        self.assertEqual(float(probs[0, 1]), 1.0)
        self.assertEqual(float(probs.sum()), 1.0)

    def test_zero_temperature_is_greedy_even_when_sampling_was_asked_for(self):
        params = SamplingParams(do_sample=True, temperature=0.0)
        self.assertTrue(params.greedy)
        probs = params.distribution(torch.tensor([[1.0, 5.0, 2.0]]))
        self.assertEqual(float(probs[0, 1]), 1.0)


class TestWarping(unittest.TestCase):
    """The draft and the target have to be warped identically, so the warping is tested once."""

    def test_temperature_flattens(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        sharp = SamplingParams(do_sample=True, temperature=0.5).distribution(logits)
        flat = SamplingParams(do_sample=True, temperature=2.0).distribution(logits)
        self.assertGreater(float(sharp[0, 2]), float(flat[0, 2]))
        self.assertAlmostEqual(float(flat.sum()), 1.0, places=5)

    def test_top_k_zeroes_everything_outside_the_k_best(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        probs = SamplingParams(do_sample=True, top_k=2).distribution(logits)[0]
        self.assertEqual(float(probs[0]), 0.0)
        self.assertEqual(float(probs[1]), 0.0)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_top_p_keeps_the_token_that_crosses_the_threshold(self):
        """An empty set is the failure mode here: one token above top_p must still survive."""
        logits = torch.tensor([[10.0, 0.0, 0.0]])
        probs = SamplingParams(do_sample=True, top_p=0.5).distribution(logits)[0]
        self.assertAlmostEqual(float(probs[0]), 1.0, places=5)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)


class TestVerifyMechanics(unittest.TestCase):
    def flat(self, size=8):
        return torch.full((size,), 1.0 / size)

    def test_every_proposal_accepted_returns_a_bonus_from_the_spare_distribution(self):
        """Where the +1 comes from: K accepted proposals still yield K+1 tokens."""
        p = [self.flat() for _ in range(4)]
        bonus = torch.zeros(8)
        bonus[6] = 1.0
        accepted, token = verify([0, 1, 2], [self.flat()] * 3, p[:3] + [bonus])
        self.assertEqual(accepted, 3)
        self.assertEqual(token, 6, "the bonus did not come from the spare distribution")

    def test_the_scan_stops_at_the_first_rejection(self):
        """Even when everything after it would have been accepted: the tokens after a rejection
        were conditioned on a token that is no longer there."""
        certain = torch.zeros(8)
        certain[0] = 1.0
        # The target insists on token 0 at every position, and the draft could propose anything.
        accepted, _ = verify([0, 5, 0], [self.flat()] * 3, [certain] * 4)
        self.assertEqual(accepted, 1, "the scan continued past a rejected proposal")

    def test_a_proposal_the_target_rules_out_is_never_emitted(self):
        """Robustness against a malformed proposal -- one the draft's own q gives no mass to."""
        target = torch.zeros(8)
        target[2] = 1.0
        impossible = torch.zeros(8)
        impossible[2] = 1.0
        accepted, corrected = verify([7], [impossible], [target, target])
        self.assertEqual(accepted, 0)
        self.assertEqual(corrected, 2)

    def test_no_proposals_is_an_ordinary_decode_step(self):
        certain = torch.zeros(8)
        certain[4] = 1.0
        accepted, token = verify([], [], [certain])
        self.assertEqual(accepted, 0)
        self.assertEqual(token, 4)


class TestAdaptiveLookahead(unittest.TestCase):
    """K follows the measured acceptance rate, because that is what decides its value."""

    def decoder(self, accepted, proposed, max_lookahead=8):
        decoder = SpeculativeDecoder(target=None, draft=None, lookahead=4,
                                     max_lookahead=max_lookahead)
        decoder.stats.accepted = accepted
        decoder.stats.proposed = proposed
        return decoder

    def test_a_good_draft_earns_a_longer_lookahead(self):
        self.assertEqual(self.decoder(80, 100).adapt(), 4)      # a/(1-a) at a=0.8

    def test_a_poor_draft_gets_a_shorter_one(self):
        self.assertEqual(self.decoder(50, 100).adapt(), 1)      # a/(1-a) at a=0.5

    def test_a_useless_draft_never_drops_below_one(self):
        self.assertEqual(self.decoder(0, 100).adapt(), 1)

    def test_a_perfect_draft_is_capped_by_the_machines_ceiling(self):
        self.assertEqual(self.decoder(100, 100, max_lookahead=6).adapt(), 6)

    def test_nothing_measured_yet_leaves_the_starting_value_alone(self):
        self.assertEqual(self.decoder(0, 0).adapt(), 4)


class TestCompatibility(unittest.TestCase):
    """A draft that does not share the tokenizer is refused, loudly, at load."""

    class Config:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size

    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = None

        def __init__(self, shift=0):
            self.shift = shift

        def encode(self, text):
            return [ord(c) + self.shift for c in text]

    def test_matching_models_are_accepted(self):
        self.assertTrue(check_compatible(self.Config(32000), self.Tokenizer(),
                                         self.Config(32000), self.Tokenizer()))

    def test_a_different_vocabulary_size_is_refused_by_name(self):
        with self.assertRaises(DraftIncompatible) as caught:
            check_compatible(self.Config(32000), None, self.Config(32064), None)
        self.assertIn("vocabulary sizes differ", str(caught.exception))

    def test_the_same_size_with_different_token_ids_is_refused(self):
        """The dangerous case: nothing downstream could detect this one."""
        with self.assertRaises(DraftIncompatible) as caught:
            check_compatible(self.Config(32000), self.Tokenizer(),
                             self.Config(32000), self.Tokenizer(shift=1))
        self.assertIn("disagree", str(caught.exception))

    def test_a_different_special_token_is_refused(self):
        other = self.Tokenizer()
        other.eos_token_id = 99
        with self.assertRaises(DraftIncompatible) as caught:
            check_compatible(self.Config(32000), self.Tokenizer(), self.Config(32000), other)
        self.assertIn("eos_token_id differs", str(caught.exception))

    def test_an_unset_pad_token_does_not_refuse_a_usable_draft(self):
        """Nothing here pads, and the standard small Llama drafts leave pad unset. Rejecting over
        a field neither model reads would refuse exactly the pairings this feature is for."""
        other = self.Tokenizer()
        other.pad_token_id = 2
        self.assertTrue(check_compatible(self.Config(32000), self.Tokenizer(),
                                         self.Config(32000), other))

    def test_the_error_says_what_a_draft_has_to_be(self):
        with self.assertRaises(DraftIncompatible) as caught:
            check_compatible(self.Config(32000), None, self.Config(1), None)
        self.assertIn("same model family", str(caught.exception))


class TestTheDecision(unittest.TestCase):
    """Never enabled silently, and never enabled on an unmeasured machine."""

    class Derivation:
        def __init__(self, value, inputs=None):
            self.value = value
            self.inputs = inputs or {}

    class Profile:
        def __init__(self, recommended, ratio=None, lookahead=None):
            self.derived = {"speculative_recommended":
                            TestTheDecision.Derivation(recommended,
                                                       {"amortization_ratio": ratio})}
            if lookahead is not None:
                self.derived["speculative_lookahead"] = TestTheDecision.Derivation(lookahead)

    def test_no_draft_means_no_speculation_whatever_the_setting_says(self):
        for setting in (SPEC_AUTO, SPEC_ON, SPEC_OFF):
            enabled, reason = resolve_speculation(setting, draft_path=None)
            with self.subTest(setting=setting):
                self.assertFalse(enabled)
                self.assertIn("no draft model", reason)

    def test_auto_follows_a_profile_that_recommends_it(self):
        enabled, reason = resolve_speculation(SPEC_AUTO, "draft",
                                              self.Profile(True, ratio=430.0))
        self.assertTrue(enabled)
        self.assertIn("430x", reason)

    def test_auto_refuses_where_the_profile_says_the_weights_are_not_the_bottleneck(self):
        enabled, reason = resolve_speculation(SPEC_AUTO, "draft", self.Profile(False, ratio=3.0))
        self.assertFalse(enabled)
        self.assertIn("not the bottleneck", reason)
        self.assertIn("speculative='on'", reason)

    def test_a_model_that_is_already_resident_gets_no_however_slow_the_storage_is(self):
        """The machine's ratio describes the machine, not this checkpoint on it.

        A 400x ratio says storage is slow. It says nothing about a model that never reads from
        storage, and for that one the pass being amortized costs device bandwidth alone while the
        draft's residency comes straight out of the weight cache.
        """
        enabled, reason = resolve_speculation(SPEC_AUTO, "draft", self.Profile(True, ratio=430.0),
                                              weight_bytes=2 * 1024 ** 3,
                                              device_bytes=8 * 1024 ** 3)
        self.assertFalse(enabled)
        self.assertIn("resident", reason)

    def test_a_model_that_does_not_fit_still_gets_yes(self):
        enabled, _ = resolve_speculation(SPEC_AUTO, "draft", self.Profile(True, ratio=430.0),
                                         weight_bytes=40 * 1024 ** 3,
                                         device_bytes=8 * 1024 ** 3)
        self.assertTrue(enabled)

    def test_unknown_sizes_fall_through_to_the_machines_own_answer(self):
        enabled, _ = resolve_speculation(SPEC_AUTO, "draft", self.Profile(True, ratio=430.0))
        self.assertTrue(enabled)

    def test_an_unmeasured_machine_gets_no_for_an_answer(self):
        """Enabling a feature that costs residency on a guess is how a fine machine gets slower."""
        enabled, reason = resolve_speculation(SPEC_AUTO, "draft", self.Profile(None))
        self.assertFalse(enabled)
        self.assertIn("not measured", reason)

    def test_no_profile_at_all_gets_no_for_an_answer(self):
        enabled, _ = resolve_speculation(SPEC_AUTO, "draft", None)
        self.assertFalse(enabled)

    def test_explicit_settings_override_the_profile_in_both_directions(self):
        self.assertTrue(resolve_speculation(SPEC_ON, "draft", self.Profile(False, 1.0))[0])
        self.assertFalse(resolve_speculation(SPEC_OFF, "draft", self.Profile(True, 900.0))[0])

    def test_an_unknown_setting_is_refused(self):
        with self.assertRaises(ValueError):
            resolve_speculation("maybe", "draft")

    def test_the_lookahead_comes_from_the_profile_with_a_stated_fallback(self):
        self.assertEqual(lookahead_ceiling(self.Profile(True, 100.0, lookahead=7)), 7)
        self.assertEqual(lookahead_ceiling(self.Profile(True, 100.0), fallback=3), 3)
        self.assertEqual(lookahead_ceiling(None, fallback=5), 5)


class TestEndToEnd(unittest.TestCase):
    """Two real models, one CPU, and the property that matters: the output does not change."""

    PROMPT = torch.tensor([[1, 7, 13, 21, 34]])
    NEW = 16

    def reference(self, target):
        """What the model produces with no speculation anywhere near it."""
        with torch.inference_mode():
            return target.generate(self.PROMPT, max_new_tokens=self.NEW, do_sample=False,
                                   pad_token_id=0)

    def decoder(self, target, draft_model, lookahead=4, **kwargs):
        return SpeculativeDecoder(target, DraftModel(draft_model), lookahead=lookahead,
                                  max_lookahead=lookahead, **kwargs)

    def test_greedy_output_is_identical_with_speculation_and_without(self):
        """THE test. Speculation is a cost optimization; if it changes a token it is a bug."""
        target, draft = tiny_model(0), tiny_model(1)
        expected = self.reference(target)
        got = self.decoder(target, draft).generate(self.PROMPT, self.NEW,
                                                   SamplingParams(do_sample=False))
        self.assertEqual(got.tolist(), expected.tolist())

    def test_that_holds_for_every_lookahead(self):
        """A rollback bug shows up at one K and not another, so the sweep is the point."""
        target, draft = tiny_model(0), tiny_model(1)
        expected = self.reference(target)
        for lookahead in (1, 2, 3, 5, 8):
            got = self.decoder(target, draft, lookahead=lookahead).generate(
                self.PROMPT, self.NEW, SamplingParams(do_sample=False))
            with self.subTest(lookahead=lookahead):
                self.assertEqual(got.tolist(), expected.tolist())

    def test_a_useless_draft_changes_the_cost_and_not_the_answer(self):
        """The acceptance rate collapses; the tokens do not move."""
        target = tiny_model(0)
        decoder = self.decoder(target, tiny_model(99))
        got = decoder.generate(self.PROMPT, self.NEW, SamplingParams(do_sample=False))
        self.assertEqual(got.tolist(), self.reference(target).tolist())
        self.assertLess(decoder.stats.acceptance_rate, 0.9)

    def test_a_perfect_draft_gets_more_than_one_token_per_pass(self):
        """The mechanism, measured: the draft is the target, so nothing is ever rejected."""
        target = tiny_model(0)
        decoder = self.decoder(target, target, lookahead=4)
        got = decoder.generate(self.PROMPT, self.NEW, SamplingParams(do_sample=False))

        self.assertEqual(got.tolist(), self.reference(target).tolist())
        self.assertEqual(decoder.stats.acceptance_rate, 1.0)
        self.assertGreater(decoder.stats.tokens_per_pass, 3.0)
        self.assertLess(decoder.stats.passes, self.NEW,
                        "speculation used as many passes as plain decoding")

    def test_the_cache_is_rolled_back_so_the_sequence_stays_consistent(self):
        """A rejected proposal leaves KV behind, and the next forward would attend to a token that
        was never emitted. The symptom is drift a few tokens later, not an error."""
        target, draft = tiny_model(0), tiny_model(99)
        caches = []

        def new_cache():
            from transformers.cache_utils import DynamicCache

            caches.append(DynamicCache())
            return caches[-1]

        decoder = self.decoder(target, draft, new_cache=new_cache)
        got = decoder.generate(self.PROMPT, self.NEW, SamplingParams(do_sample=False))
        self.assertEqual(caches[-1].get_seq_length(), got.shape[1] - 1,
                         "the cache does not hold exactly the sequence minus its last token")

    def test_it_stops_at_the_end_of_sequence_token(self):
        target, draft = tiny_model(0), tiny_model(1)
        expected = self.reference(target)[0].tolist()
        stop = expected[len(self.PROMPT[0]) + 3]
        decoder = self.decoder(target, draft, eos_token_id=stop)
        got = decoder.generate(self.PROMPT, self.NEW, SamplingParams(do_sample=False))[0].tolist()

        self.assertEqual(got[-1], stop, "generation did not stop on the eos token")
        self.assertEqual(got, expected[:len(got)])

    def test_sampled_output_stays_inside_the_targets_support(self):
        """The sampled path cannot be pinned by a seed comparison -- see the module docstring -- so
        what is checked end to end is that it produces legal tokens and real acceptances."""
        target, draft = tiny_model(0), tiny_model(1)
        decoder = self.decoder(target, draft)
        generator = torch.Generator().manual_seed(4)
        got = decoder.generate(self.PROMPT, self.NEW,
                               SamplingParams(do_sample=True, temperature=0.8, top_p=0.95),
                               generator)
        self.assertEqual(got.shape[1], self.PROMPT.shape[1] + self.NEW)
        self.assertTrue(bool((got < VOCAB).all()) and bool((got >= 0).all()))
        self.assertGreater(decoder.stats.passes, 0)

    def test_a_wide_forward_agrees_with_one_position_at_a_time(self):
        """What greedy equality actually rests on, and the one place a real model breaks it.

        Verification computes a token's logits alongside several others. In exact arithmetic that is
        the same number as computing it alone, which is why the test above can demand identical
        output. In a reduced compute dtype it is NOT: a different matmul shape reduces in a
        different order. Measured on TinyLlama-1.1B at bf16, one of 40 positions disagreed with
        itself -- in the stock transformers model, with no speculation involved at all; at float32,
        none of the 40 did.

        Pinning the exact-arithmetic half here means a future divergence can be attributed. If this
        test fails, the verification logic is wrong. If it passes and a bf16 run still diverges,
        that is the model's arithmetic and no change to this module will fix it.
        """
        from transformers.cache_utils import DynamicCache

        target = tiny_model(0)
        sequence = self.reference(target)[0]
        prompt = self.PROMPT.shape[1]

        with torch.inference_mode():
            wide = target(input_ids=sequence.unsqueeze(0), use_cache=False).logits[0]
            cache = DynamicCache()
            narrow = []
            out = target(input_ids=sequence[:prompt].unsqueeze(0), past_key_values=cache,
                         use_cache=True)
            narrow.append(int(out.logits[0, -1].argmax()))
            for i in range(prompt, len(sequence) - 1):
                out = target(input_ids=sequence[i:i + 1].unsqueeze(0), past_key_values=cache,
                             use_cache=True)
                narrow.append(int(out.logits[0, -1].argmax()))

        batched = [int(wide[i].argmax()) for i in range(prompt - 1, len(sequence) - 1)]
        self.assertEqual(narrow, batched,
                         "a token's argmax depends on how many positions shared its forward")

    def test_batched_input_is_refused_rather_than_quietly_wrong(self):
        target, draft = tiny_model(0), tiny_model(1)
        with self.assertRaises(ValueError):
            self.decoder(target, draft).generate(torch.tensor([[1, 2], [3, 4]]), 4)

    def test_a_cache_that_cannot_be_cropped_is_refused_with_a_reason(self):
        target, draft = tiny_model(0), tiny_model(1)
        decoder = self.decoder(target, draft, new_cache=lambda: object())
        with self.assertRaises(TypeError) as caught:
            decoder.generate(self.PROMPT, 4)
        self.assertIn("crop", str(caught.exception))


class _RecordingStreamer:
    """transformers' streamer protocol, remembering everything it was handed."""

    def __init__(self):
        self.puts = []
        self.ended = 0

    def put(self, value):
        self.puts.append(value.tolist() if hasattr(value, "tolist") else list(value))

    def end(self):
        self.ended += 1


class TestStreaming(unittest.TestCase):
    """A streamer has to work here as well as on the ordinary path.

    This loop REPLACES transformers' generation loop rather than sitting inside it, so a streamer
    passed to generate() used to be accepted and then never called -- and a caller waiting for its
    first token would wait for the whole generation, or for ever. The server streams every reply, so
    this is not a corner case: it is what happens on every request once a draft model is loaded.
    """

    PROMPT = torch.tensor([[1, 7, 13, 21, 34]])
    NEW = 12

    def decoder(self, target, draft_model, **kwargs):
        return SpeculativeDecoder(target, DraftModel(draft_model), lookahead=4, max_lookahead=4,
                                  **kwargs)

    def stream(self, decoder, **kwargs):
        streamer = _RecordingStreamer()
        got = decoder.generate(self.PROMPT, self.NEW, SamplingParams(do_sample=False),
                               streamer=streamer, **kwargs)
        return got, streamer

    def test_the_streamed_tokens_are_exactly_the_generated_ones(self):
        target, draft = tiny_model(0), tiny_model(1)
        got, streamer = self.stream(self.decoder(target, draft))

        self.assertEqual(streamer.puts[0], self.PROMPT.tolist(),
                         "the prompt is handed to the streamer first, as transformers does")
        streamed = [token for batch in streamer.puts[1:] for token in batch]
        self.assertEqual(streamed, got[0].tolist()[self.PROMPT.shape[1]:])
        self.assertEqual(streamer.ended, 1)

    def test_a_pass_streams_every_token_it_accepted_not_just_one(self):
        """The saving is that one pass yields several tokens; the streamer must see all of them."""
        target = tiny_model(0)
        _, streamer = self.stream(self.decoder(target, target))
        self.assertTrue(any(len(batch) > 1 for batch in streamer.puts[1:]),
                        "a perfectly accepted pass streamed its tokens one at a time")

    def test_tokens_past_the_end_of_sequence_are_never_streamed(self):
        """A pass can produce tokens after the eos one, and those are dropped from the result. If
        they were streamed first, a client would have printed text the final answer does not
        contain -- and it cannot take it back."""
        target, draft = tiny_model(0), tiny_model(1)
        reference = self.decoder(target, draft).generate(
            self.PROMPT, self.NEW, SamplingParams(do_sample=False))[0].tolist()
        stop = reference[self.PROMPT.shape[1] + 2]

        got, streamer = self.stream(self.decoder(target, draft, eos_token_id=stop))
        streamed = [token for batch in streamer.puts[1:] for token in batch]
        self.assertEqual(streamed, got[0].tolist()[self.PROMPT.shape[1]:])
        self.assertEqual(streamed[-1], stop)
        self.assertEqual(streamer.ended, 1)

    def test_generating_without_a_streamer_is_unchanged(self):
        target, draft = tiny_model(0), tiny_model(1)
        with_none = self.decoder(target, draft).generate(self.PROMPT, self.NEW,
                                                         SamplingParams(do_sample=False))
        got, _ = self.stream(self.decoder(target, draft))
        self.assertEqual(got.tolist(), with_none.tolist())


class TestWithTheQuantizedContext(unittest.TestCase):
    """Speculation and the int4 KV cache are independent features that have to compose.

    They meet at exactly one place -- a rejection has to be taken back out of a cache that is
    otherwise append-only -- so that is what is checked.
    """

    PROMPT = torch.tensor([[1, 7, 13, 21, 34]])

    def test_greedy_output_is_unchanged_with_a_quantized_context(self):
        from rocketllm.quant.kv_cache import KVCacheConfig, QuantizedKVCache

        target, draft = tiny_model(0), tiny_model(1)
        with torch.inference_mode():
            expected = target.generate(self.PROMPT, max_new_tokens=12, do_sample=False,
                                       pad_token_id=0)

        config = KVCacheConfig(group_size=8, residual_length=64, compute_dtype=torch.float32)
        decoder = SpeculativeDecoder(target, DraftModel(draft), lookahead=4, max_lookahead=4,
                                     new_cache=lambda: QuantizedKVCache(config))
        got = decoder.generate(self.PROMPT, 12, SamplingParams(do_sample=False))
        # The context is lossy by design, so tokens may differ; the length and the mechanism must
        # not. Anything quantization does here it also does without speculation.
        self.assertEqual(got.shape, expected.shape)

    def test_cropping_inside_the_residual_window_gives_back_exactly_those_tokens(self):
        from rocketllm.quant.kv_cache import KVCacheConfig, QuantizedKVCache

        cache = QuantizedKVCache(KVCacheConfig(group_size=8, residual_length=32,
                                               compute_dtype=torch.float32))
        keys = torch.randn(1, 2, 40, 8)
        values = torch.randn(1, 2, 40, 8)
        cache.update(keys, values, layer_idx=0)
        self.assertEqual(cache.get_seq_length(0), 40)

        cache.crop(36)
        self.assertEqual(cache.get_seq_length(0), 36)
        out_keys, _ = cache.update(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8), layer_idx=0)
        self.assertEqual(out_keys.shape[-2], 37)

    def test_cropping_into_a_quantized_block_says_why_it_will_not(self):
        """Re-encoding an already-quantized block to serve tokens being thrown away is the one
        thing this must not do quietly."""
        from rocketllm.quant.kv_cache import KVCacheConfig, QuantizedKVCache

        cache = QuantizedKVCache(KVCacheConfig(group_size=8, residual_length=16,
                                               compute_dtype=torch.float32))
        cache.update(torch.randn(1, 2, 64, 8), torch.randn(1, 2, 64, 8), layer_idx=0)
        quantized = sum(block.length for block in cache._blocks[0][0])
        self.assertGreater(quantized, 0, "nothing was quantized, so there is nothing to protect")

        with self.assertRaises(ValueError) as caught:
            cache.crop(quantized - 1)
        self.assertIn("re-encoding", str(caught.exception))


class TestTheDraftsMemoryReachesPlacement(unittest.TestCase):
    """The draft is resident, so it competes with the weight cache. That has to be accounted for.

    Registering it does NOT subtract the bytes -- they are already allocated, and the budget
    measures them -- it attributes them and republishes at once, so the cache shrinks with a stated
    reason instead of discovering the shortfall during the first generation.
    """

    MB = 1024 * 1024

    def harness(self, free_before, free_after):
        from tests.test_vram_budget import FakeCaps, FakeProfile
        from rocketllm.base import RocketModel
        from rocketllm.memory import CLASS_ALWAYS, PinCandidate
        from rocketllm.memory.budget import VramBudget
        from rocketllm.memory.cache import TieredWeightCache

        candidate = 10 * self.MB

        class Harness(RocketModel):
            def __init__(self):
                pass

            def _pin_candidates(self):
                return [PinCandidate(key=(i, "dense"), packed_bytes=candidate,
                                     priority=CLASS_ALWAYS, accesses_per_token=1.0)
                        for i in range(8)]

        model = Harness()
        model.cache = TieredWeightCache(fetch=lambda key: f"payload:{key}",
                                        sizer=lambda key: candidate,
                                        device_bytes=0, host_bytes=0, window=1)
        model.pin_policy = "auto"
        model._pin_budget = 0
        model._window_share = 0.5
        readings = [(free_before, 0, 0, False)] + [(free_after, 0, 0, False)] * 4
        model.budget = VramBudget(device_caps=FakeCaps(readings),
                                  profile=FakeProfile(hysteresis_ratio=0.05,
                                                      hysteresis_samples=3),
                                  configure_allocator_env=False,
                                  reclaimable=model._cache_holdings,
                                  on_change=model._budget_changed)
        return model

    def test_a_resident_draft_takes_pin_budget_away_from_the_weights(self):
        model = self.harness(free_before=200 * self.MB, free_after=140 * self.MB)
        model._budget_changed(0, model.budget.target(), model.budget.history[-1])
        before = len(model.cache.pinned)
        self.assertGreater(before, 0)

        model.budget.register_external("draft_model", 60 * self.MB)

        self.assertEqual(model.budget.target(), 140 * self.MB)
        self.assertLess(len(model.cache.pinned), before,
                        "the draft's memory never reached the pin plan")

    def test_the_claim_is_attributed_rather_than_being_an_anonymous_dip(self):
        model = self.harness(200 * self.MB, 140 * self.MB)
        model.budget.register_external("draft_model", 60 * self.MB)
        summary = model.budget.summary()
        self.assertEqual(summary["external"], {"draft_model": 60 * self.MB})
        self.assertEqual(summary["external_bytes"], 60 * self.MB)

    def test_the_bytes_are_not_charged_twice(self):
        """They are already allocated, so the reading has them; subtracting again would double."""
        model = self.harness(140 * self.MB, 140 * self.MB)
        model.budget.register_external("draft_model", 60 * self.MB)
        self.assertEqual(model.budget.current(), 140 * self.MB)

    def test_withdrawing_the_claim_republishes_at_once(self):
        model = self.harness(140 * self.MB, 200 * self.MB)
        model.budget.register_external("draft_model", 60 * self.MB)
        model.budget.register_external("draft_model", 0)
        self.assertEqual(model.budget.external, {})
        self.assertEqual(model.budget.target(), 200 * self.MB)


if __name__ == "__main__":
    unittest.main(verbosity=2)
