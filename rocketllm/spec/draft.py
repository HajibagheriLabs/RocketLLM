"""Speculative decoding: a small resident model proposes, one big pass verifies.

Everywhere else in this engine the unit of cost is the STREAMING PASS -- one walk through the
weights, storage -> host -> device. A pass costs the same whether it produces one token or twenty,
because what it pays for is moving the bytes, not computing with them. Ordinary decoding buys
exactly one token per pass, which is the worst possible deal on a machine where the weights do not
fit.

Speculation changes the exchange rate. A small model that IS resident proposes K tokens for
essentially nothing, and a single pass of the big model checks all K at once. Accept a of them and
the pass produced a+1 tokens instead of 1.

So the size of the win is set by which tier the bytes come from, and it is not a constant factor:

  * storage-bound (weights far exceed VRAM+RAM)  -- a pass costs hundreds of milliseconds and the
    draft costs nothing next to it. This is where speculation is worth having.
  * resident (the model fits on the device)      -- a pass is already cheap, the draft's own
    forwards are a real fraction of it, and the memory the draft occupies comes straight out of the
    weight cache. It can be, and often is, NET NEGATIVE.

Which is why this is never switched on silently: the default comes from the hardware profile's
measured amortization ratio, and where the profile says no, no is what happens.

Correctness
-----------
The algorithm is exactly distribution-preserving, and that is a claim about arithmetic rather than
about tokens. For a proposal x drawn from the draft's q, accepted with probability min(1, p(x)/q(x))
and otherwise replaced by a draw from the normalized residual (p - q)+, the token that comes out is
distributed as p:

    P(t) = min(q(t), p(t)) + (1 - sum_x min(q(x), p(x))) * (p(t) - q(t))+ / sum_x (p(x) - q(x))+
         = min(q(t), p(t)) + (p(t) - q(t))+                        [the two sums are equal]
         = p(t)

Greedy is the same rule at zero temperature, where p is a point mass: the ratio test reduces to
"accept iff the draft guessed the argmax", and the residual reduces to a point mass on it. Both fall
out of the general path rather than being special-cased, so there is one implementation to get right.

The tests assert that identity three ways: algebraically at machine precision, empirically over many
draws, and end to end by requiring greedy output with speculation on to equal greedy output with it
off, token for token. The sampled path CANNOT be asserted the same last way -- rejection sampling
consumes randomness that direct sampling does not, so under any fixed seed the two produce different
draws from the same distribution. Asserting stream equality there would be asserting something false
about the algorithm; asserting the distribution is asserting the thing that is actually true.

The one caveat, and it is not this module's
-------------------------------------------
All of the above is exact arithmetic. In a reduced compute dtype the equality is not exact, and the
reason has nothing to do with speculation: a token's logits are not bitwise identical when computed
inside a wide forward instead of alone, because a different matmul shape reduces in a different
order. Measured on TinyLlama-1.1B at bf16, taking the stock transformers model with no speculation
anywhere near it and computing 40 positions both ways, ONE argmax disagreed -- 'Germany' decoded one
at a time, 'France' decoded in a batch. At float32, none of the 40 disagreed.

Verification is a wide forward by construction, so it inherits that. Driving this decoder against
stock transformers models on the same prompt, greedy speculation reproduced plain greedy decoding
token for token at float32 and diverged at token 14 at bf16 -- same code, same draft, same seed, one
variable. Where a target's top two tokens are within bf16 noise of each other, greedy speculation
can take the other one, and the run continues from a different token entirely.

Nothing here can prevent that and no implementation of this algorithm on any framework can; it is a
property of the model's arithmetic, not of the accept test. It is written down because the
alternative is someone later reading a diverged greedy run as a bug in this file and "fixing" the
accept test until it goes away.
"""
import dataclasses
import logging
import time

import torch

from ..hw.caps import announce_once

log = logging.getLogger(__name__)

#: Settings the engine accepts. "auto" defers to the hardware profile's recommendation.
SPEC_AUTO = "auto"
SPEC_ON = "on"
SPEC_OFF = "off"
SPEC_CHOICES = (SPEC_AUTO, SPEC_ON, SPEC_OFF)

#: Strings the tokenizers are asked to encode when checking they agree. Chosen to exercise what
#: differs between two tokenizers that share a vocabulary size: byte fallback, digit splitting,
#: whitespace handling and punctuation.
_PROBES = (
    "The capital of France is Paris.",
    "  leading and trailing whitespace  ",
    "1234567890 3.14159",
    "def f(x):\n\treturn x ** 2\n",
    "punctuation -- dashes, quotes and 'apostrophes'",
)


class DraftIncompatible(ValueError):
    """The draft cannot stand in for the target: different vocabulary, or a different tokenizer.

    Raised at load, deliberately, rather than letting the run start. Two models with mismatched
    vocabularies still produce a perfectly well-formed accept/reject decision -- on token ids that
    mean different things in each -- so the failure mode without this check is not a crash. It is
    fluent output that is quietly wrong, at a low acceptance rate that looks like an ordinary bad
    draft.
    """


def check_compatible(target_config, target_tokenizer, draft_config, draft_tokenizer):
    """Confirm the draft speaks exactly the same token language as the target, or say what differs.

    Vocabulary size alone is not enough, and that is the trap here: two models can both report 32000
    and disagree about which id is which, which nothing downstream can detect. So the tokenizers are
    also asked to encode a handful of strings and their answers compared, which settles merges,
    normalization and special-token placement in one cheap check.
    """
    problems = []

    target_vocab = getattr(target_config, "vocab_size", None)
    draft_vocab = getattr(draft_config, "vocab_size", None)
    if target_vocab and draft_vocab and int(target_vocab) != int(draft_vocab):
        problems.append(f"vocabulary sizes differ: target has {target_vocab}, draft has "
                        f"{draft_vocab}. The two models' logits are not over the same set of "
                        f"tokens, so no acceptance test between them means anything")

    if target_tokenizer is not None and draft_tokenizer is not None:
        # bos and eos only. Both are part of the token language: bos changes how a prompt encodes
        # and eos is what stops a generation. pad is deliberately NOT checked -- nothing here pads,
        # batch-of-one decoding never sees it, and the standard small Llama drafts leave it unset
        # while agreeing on every token that exists. Refusing over it would reject the exact
        # pairings this feature is for, over a field neither model reads.
        for name in ("bos_token_id", "eos_token_id"):
            mine = getattr(target_tokenizer, name, None)
            theirs = getattr(draft_tokenizer, name, None)
            if mine != theirs:
                problems.append(f"{name} differs: target {mine!r}, draft {theirs!r}")
        for probe in _PROBES:
            mine = target_tokenizer.encode(probe)
            theirs = draft_tokenizer.encode(probe)
            if mine != theirs:
                problems.append(
                    f"the tokenizers disagree on {probe!r}: the target encodes it as {mine}, the "
                    f"draft as {theirs}. Same-sized vocabularies with different token ids are the "
                    f"dangerous case -- the run would look fine and be wrong")
                break

    if problems:
        raise DraftIncompatible(
            "the draft model cannot verify against this target:\n  - "
            + "\n  - ".join(problems)
            + "\nA draft has to share the target's tokenizer, which in practice means the smallest "
              "member of the same model family.")
    return True


# -- turning logits into the distribution that is actually being sampled ------------------------

@dataclasses.dataclass(frozen=True)
class SamplingParams:
    """The warping applied to logits before anything is drawn from them.

    The draft and the target MUST be warped identically. The distribution speculation preserves is
    whatever is actually sampled from -- if top_p truncates the target's tail then the preserved
    distribution is the truncated one, which is correct, but only if q is truncated the same way.
    Warping one and not the other produces a low acceptance rate rather than an error, so the two
    share this object instead of each holding their own.
    """

    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    @classmethod
    def from_generation(cls, config=None, **overrides):
        """Read the effective settings out of a GenerationConfig plus per-call overrides."""
        def pick(name, fallback):
            value = overrides.get(name)
            if value is None and config is not None:
                value = getattr(config, name, None)
            return fallback if value is None else value

        return cls(do_sample=bool(pick("do_sample", False)),
                   temperature=float(pick("temperature", 1.0) or 1.0),
                   top_k=int(pick("top_k", 0) or 0),
                   top_p=float(pick("top_p", 1.0) or 1.0))

    @property
    def greedy(self):
        # Zero temperature is greedy however do_sample was set: it is a point mass either way, and
        # dividing by it would be a crash rather than a decision.
        return not self.do_sample or self.temperature <= 0.0

    def distribution(self, logits):
        """Logits -> the probability vector that is sampled from. Shape [..., vocab].

        Greedy returns a one-hot rather than taking a different branch through the algorithm. The
        acceptance test against a point mass is exactly "did the draft guess the argmax", and the
        residual on rejection is exactly that argmax, so greedy does not need a separate
        implementation. It needs an honest p.
        """
        logits = logits.float()
        if self.greedy:
            out = torch.zeros_like(logits)
            out.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
            return out

        logits = logits / self.temperature
        if self.top_k and 0 < self.top_k < logits.shape[-1]:
            cutoff = torch.topk(logits, self.top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        if 0.0 < self.top_p < 1.0:
            ordered, index = torch.sort(probs, descending=True, dim=-1)
            cumulative = ordered.cumsum(dim=-1)
            # Keep the token that crosses the threshold, so the kept mass is at least top_p and the
            # set is never empty even when one token already exceeds it on its own.
            ordered = ordered.masked_fill(cumulative - ordered >= self.top_p, 0.0)
            probs = torch.zeros_like(probs).scatter_(-1, index, ordered)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs


def draw(probs, generator=None):
    """One token id from a probability vector of shape [vocab].

    A point mass draws deterministically, which is what makes the greedy path fall out of the same
    code. The generator carries its own device and torch will not mix the two, so the probabilities
    come to it rather than the other way round.
    """
    if generator is not None and probs.device != generator.device:
        probs = probs.to(generator.device)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def _uniform(generator):
    device = generator.device if generator is not None else "cpu"
    return float(torch.rand((), generator=generator, device=device))


# -- the draft ------------------------------------------------------------------------------------

class DraftModel:
    """A small dense model, fully resident, that proposes tokens.

    Fully resident is the whole point and is not negotiable: a draft that had to be streamed would
    cost a pass of its own per proposal and there would be nothing left to amortize. It is also why
    its memory is registered with the device budget -- those bytes come out of the weight cache, and
    the honest accounting is that keeping a draft is a placement decision like any other.
    """

    def __init__(self, model, tokenizer=None, device=None, dtype=None, name=""):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name or getattr(getattr(model, "config", None), "name_or_path", "draft")
        self.device = device
        self.dtype = dtype
        self.cache = None
        self.forwards = 0
        self.seconds = 0.0

    @classmethod
    def load(cls, path, target_config=None, target_tokenizer=None, device=None, dtype=None,
             hf_token=None, trust_remote_code=False):
        """Load a draft from a repo id or local path, and refuse it if it cannot verify.

        The compatibility check runs BEFORE the weights are placed, so a mismatched draft costs a
        config and a tokenizer rather than a download and a device allocation.
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        token_kwargs = {"token": hf_token} if hf_token is not None else {}
        config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code,
                                            **token_kwargs)
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code,
                                                      **token_kwargs)
        except Exception as exc:  # noqa: BLE001 - a missing tokenizer weakens the check, not the load
            log.debug("the draft at %s has no loadable tokenizer (%s); checking the config only",
                      path, exc)
        if target_config is not None:
            check_compatible(target_config, target_tokenizer, config, tokenizer)

        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, trust_remote_code=trust_remote_code, **token_kwargs)
        model.eval()
        if device is not None:
            model.to(device)
        return cls(model, tokenizer=tokenizer, device=device, dtype=dtype, name=str(path))

    def device_bytes(self):
        """Device memory the draft occupies, summed from its own tensors.

        Summed rather than taken from an allocator delta so it means the same thing on every
        backend, and so it stays right when something else allocates during the load.
        """
        total = 0
        for tensor in list(self.model.parameters()) + list(self.model.buffers()):
            if tensor is not None and tensor.device.type != "meta":
                total += tensor.numel() * tensor.element_size()
        return total

    # -- proposing ---------------------------------------------------------------------------

    def reset(self):
        self.cache = None

    def crop(self, length):
        """Drop cached positions past `length`, after a rejection took some proposals back."""
        if self.cache is not None and length >= 0:
            self.cache.crop(length)

    def propose(self, sequence, cached, k, sampling, generator=None):
        """Extend `sequence` by `k` tokens, one forward each, and report the distributions used.

        Returns ``(tokens, probs, cached)``: `probs[i]` is the full distribution `tokens[i]` was
        drawn from, which the verifier needs in order to correct a rejection, and `cached` is how
        many positions the draft's own cache holds afterwards. The last proposal is deliberately not
        fed back through -- nothing needs its successor's distribution, and skipping it saves a
        forward on every pass.
        """
        if self.cache is None:
            self.cache = _dynamic_cache()

        tokens, probs = [], []
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(k):
                chunk = sequence[:, cached:]
                if chunk.shape[1] == 0:
                    break
                logits = self.model(input_ids=chunk, past_key_values=self.cache,
                                    use_cache=True).logits[:, -1, :]
                cached += chunk.shape[1]
                self.forwards += 1
                distribution = sampling.distribution(logits)[0]
                token = draw(distribution, generator)
                tokens.append(token)
                probs.append(distribution)
                sequence = torch.cat(
                    [sequence, torch.tensor([[token]], dtype=sequence.dtype,
                                            device=sequence.device)], dim=1)
        self.seconds += time.perf_counter() - started
        return tokens, probs, cached


# -- the algorithm ---------------------------------------------------------------------------------

@dataclasses.dataclass
class SpeculationStats:
    """What speculation actually did, which is the only way to know whether it was worth it."""

    passes: int = 0
    proposed: int = 0
    accepted: int = 0
    emitted: int = 0
    target_seconds: float = 0.0
    draft_seconds: float = 0.0
    lookahead_sum: int = 0

    @property
    def acceptance_rate(self):
        """Share of proposed tokens the target kept. The number the whole trade rests on."""
        return (self.accepted / self.proposed) if self.proposed else 0.0

    @property
    def tokens_per_pass(self):
        """Effective tokens produced per verification pass. At 1.0 speculation bought nothing."""
        return (self.emitted / self.passes) if self.passes else 0.0

    @property
    def mean_lookahead(self):
        return (self.lookahead_sum / self.passes) if self.passes else 0.0

    def to_dict(self):
        return {
            "passes": self.passes,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "emitted": self.emitted,
            "acceptance_rate": self.acceptance_rate,
            "tokens_per_pass": self.tokens_per_pass,
            "mean_lookahead": self.mean_lookahead,
            "target_seconds": self.target_seconds,
            "draft_seconds": self.draft_seconds,
        }


def verify(proposals, draft_probs, target_probs, generator=None):
    """The accept/reject core. Returns ``(accepted_count, next_token)``.

    `target_probs[i]` is the target's distribution for the position `proposals[i]` occupies, and
    there is one more of them than there are proposals. That spare is where the bonus token comes
    from when every proposal survives, and it is the "+1" in the speedup: a fully accepted pass
    returns K+1 tokens, not K.

    Deliberately free of models, caches and devices: this is the part that is easy to get subtly
    wrong, and it is worth being able to check on paper.
    """
    accepted = 0
    for index, token in enumerate(proposals):
        p = target_probs[index]
        q = draft_probs[index]
        p_token = float(p[token])
        q_token = float(q[token])
        if q_token > 0.0:
            ratio = p_token / q_token
        else:
            # A proposal the draft gives no mass to cannot have been drawn from it, so this is a
            # caller error rather than a case of the algorithm. The limit of min(1, p/q) says
            # accept, and for a legal input that costs nothing because such a token is never
            # proposed -- but only where the target allows it. Emitting a token the target rules
            # out entirely, on the strength of a malformed proposal, is the one outcome worth
            # refusing outright.
            ratio = 1.0 if p_token > 0.0 else 0.0
        if ratio >= 1.0:
            accepted += 1
            continue
        # No draw when the target gives the proposal no mass at all: the outcome is settled, and
        # consuming randomness to learn that would make greedy runs depend on the seed.
        if ratio > 0.0 and _uniform(generator) < ratio:
            accepted += 1
            continue
        return accepted, _resample(p, q, generator)
    return accepted, draw(target_probs[len(proposals)], generator)


def _resample(p, q, generator):
    """Draw the replacement for a rejected proposal, from the corrected distribution (p - q)+.

    This is the half that makes the whole thing exact. Rejecting a proposal removes mass from p in
    proportion to q, and drawing the replacement from the positive part of the difference puts
    exactly that mass back -- see the identity in the module docstring. Drawing it from p instead,
    which is the natural-looking mistake, biases the output towards whatever the draft is bad at.
    """
    residual = torch.clamp(p - q, min=0.0)
    total = float(residual.sum())
    if total <= 0.0:
        # Only reachable through floating-point cancellation, since a rejection means p and q differ
        # somewhere. Falling back to p keeps the draw legal rather than dividing by zero.
        return draw(p, generator)
    return draw(residual / total, generator)


class SpeculativeDecoder:
    """Runs one generation: draft K, verify in a single target pass, repeat.

    The loop's invariant is that the target cache holds every token except the last one emitted.
    That last token is fed to the target together with the proposals, so a single pass produces the
    distribution for every position that needs checking plus one spare -- which is what lets a fully
    accepted run return K+1 tokens from one pass.

    `target` is anything callable as ``target(input_ids=..., past_key_values=..., use_cache=True)``
    returning ``.logits``. That is deliberately the plain transformers signature: the streaming
    engine, a full-precision reference and a two-layer test model are all the same thing here.
    """

    def __init__(self, target, draft, lookahead=4, max_lookahead=8, new_cache=None,
                 eos_token_id=None, on_pass=None):
        self.target = target
        self.draft = draft
        self.max_lookahead = max(1, int(max_lookahead))
        self.lookahead = max(1, min(int(lookahead), self.max_lookahead))
        self.new_cache = new_cache
        self.eos_token_id = _as_set(eos_token_id)
        #: Called with the tokens one pass produced. The benchmark harness rides on this to split
        #: prefill from decode, which it cannot do through a logits processor here: transformers'
        #: generation loop is not the loop being run.
        self.on_pass = on_pass
        self.stats = SpeculationStats()

    def adapt(self):
        """Set the next pass's K from the acceptance rate measured so far.

        Under an acceptance rate a, the number of proposals that survive before the first rejection
        is geometric with mean a/(1-a). Proposing about that many is the natural target: fewer
        leaves accepted tokens unclaimed, more spends draft forwards on proposals that will be
        thrown away. It needs no tuning constant, because the quantity it is derived from is
        measured, and it is bounded above by what the profile says a pass on this machine is worth
        waiting for.
        """
        if not self.stats.proposed:
            return self.lookahead
        rate = self.stats.acceptance_rate
        expected = float(self.max_lookahead) if rate >= 1.0 else rate / (1.0 - rate)
        self.lookahead = max(1, min(self.max_lookahead, int(round(expected))))
        return self.lookahead

    def generate(self, input_ids, max_new_tokens, sampling=None, generator=None, streamer=None):
        """Generate up to `max_new_tokens`, returning the whole sequence including the prompt.

        `streamer` follows transformers' streamer protocol exactly -- the prompt first, then each
        pass's tokens, then ``end()`` -- so a caller cannot tell which loop produced them. That
        matters because this loop REPLACES transformers' generation loop rather than sitting inside
        it: a streamer handed to generate() would otherwise be accepted and silently never called,
        and the caller would wait forever for a first token.
        """
        sampling = sampling or SamplingParams()
        if input_ids.shape[0] != 1:
            raise ValueError("speculative decoding runs one sequence at a time; batch outside it, "
                             "or turn speculation off for batched generation")

        cache = self.new_cache() if self.new_cache is not None else _dynamic_cache()
        if not hasattr(cache, "crop"):
            raise TypeError(f"{type(cache).__name__} cannot be cropped, and a rejected proposal has "
                            f"to come back out of the cache. Speculation needs a cache that "
                            f"implements crop().")
        self.draft.reset()

        if streamer is not None:
            streamer.put(input_ids)

        sequence = input_ids
        prompt_length = sequence.shape[1]
        cached = 0            # positions the TARGET cache holds
        draft_cached = 0      # positions the DRAFT cache holds
        emitted = 0

        while emitted < max_new_tokens:
            # One fewer proposal than there is room for: the pass yields a token of its own even
            # with nothing proposed, so the last token of a run never needs a draft forward.
            k = max(0, min(self.lookahead, max_new_tokens - emitted - 1))
            proposals, draft_probs, draft_cached = self.draft.propose(
                sequence, draft_cached, k, sampling, generator)

            pending = sequence[:, cached:]
            candidate = torch.cat(
                [pending, torch.tensor([proposals], dtype=sequence.dtype,
                                       device=sequence.device)], dim=1) if proposals else pending

            started = time.perf_counter()
            with torch.inference_mode():
                logits = self.target(input_ids=candidate, past_key_values=cache,
                                     use_cache=True).logits
            self.stats.target_seconds += time.perf_counter() - started
            self.stats.passes += 1
            self.stats.lookahead_sum += len(proposals)

            # One distribution per proposal, plus the spare the bonus token comes from.
            width = len(proposals) + 1
            target_probs = [sampling.distribution(logits[:, i - width, :])[0] for i in range(width)]
            accepted, corrected = verify(proposals, draft_probs, target_probs, generator)
            self.stats.proposed += len(proposals)
            self.stats.accepted += accepted

            new_tokens = (proposals[:accepted] + [corrected])[:max_new_tokens - emitted]
            # Where this pass ends, if it does. Taken BEFORE anything is streamed, because a pass
            # can produce tokens after the end-of-sequence one and those are dropped from the
            # returned sequence -- streaming them first would show a client text that the final
            # answer does not contain.
            ends_at = next((i for i, t in enumerate(new_tokens) if t in self.eos_token_id),
                           None) if self.eos_token_id else None
            if streamer is not None:
                streamer.put(new_tokens if ends_at is None else new_tokens[:ends_at + 1])
            sequence = torch.cat(
                [sequence, torch.tensor([new_tokens], dtype=sequence.dtype,
                                        device=sequence.device)], dim=1)
            emitted += len(new_tokens)
            self.stats.emitted += len(new_tokens)
            self.stats.draft_seconds = self.draft.seconds
            if self.on_pass is not None:
                self.on_pass(len(new_tokens))

            # The pass wrote KV for every candidate position, including the proposals past the
            # accepted prefix. Those are now positions of tokens that were never emitted, and they
            # have to come back out or the next forward attends to them.
            cached += pending.shape[1] + accepted
            cache.crop(cached)
            draft_cached = min(draft_cached, cached)
            self.draft.crop(draft_cached)

            if ends_at is not None:
                keep = prompt_length + emitted - len(new_tokens) + ends_at + 1
                if streamer is not None:
                    streamer.end()
                return sequence[:, :keep]
            self.adapt()

        if streamer is not None:
            streamer.end()
        return sequence


def _dynamic_cache():
    from transformers.cache_utils import DynamicCache

    return DynamicCache()


def _as_set(eos_token_id):
    if eos_token_id is None:
        return frozenset()
    if isinstance(eos_token_id, int):
        return frozenset({eos_token_id})
    return frozenset(int(token) for token in eos_token_id)


# -- whether to do it at all ----------------------------------------------------------------------

def resolve_speculation(setting, draft_path=None, profile=None, weight_bytes=None,
                        device_bytes=None):
    """Turn a `speculative=` setting into a decision, and say why. Returns ``(enabled, reason)``.

    Two things have to be true for speculation to pay, and they are questions about different
    objects, so both are asked.

    The first is about the MACHINE: does moving the weights dominate computing with them? That is
    the profile's measured amortization ratio, device memory bandwidth over the slowest tier that
    has to serve weights. Where it was not measured the answer is no -- enabling a feature that
    takes device memory away from the weight cache on an unmeasured guess is how a machine that was
    fine gets slower for no stated reason.

    The second is about this MODEL on that machine: are the weights actually crossing that slow
    tier? A ratio of 400x describes a machine whose storage is slow, and it says nothing at all
    about a checkpoint that is already resident -- for that one, decoding never touches storage, the
    pass being amortized costs device bandwidth alone, and the draft's residency comes straight out
    of the weight cache for nothing. The same card is storage-bound for one checkpoint and roomy for
    the next, which is why the profile alone cannot answer this and why the sizes are passed in.
    """
    if setting not in SPEC_CHOICES:
        raise ValueError(f"speculative must be one of {', '.join(SPEC_CHOICES)}, not {setting!r}")
    if not draft_path:
        return False, "no draft model was given, so there is nothing to propose with"
    if setting == SPEC_OFF:
        return False, "turned off explicitly"
    if setting == SPEC_ON:
        return True, "turned on explicitly"

    if weight_bytes and device_bytes and weight_bytes <= device_bytes:
        return False, (f"the weights are resident ({weight_bytes / 1024 ** 3:.1f}GB in a "
                       f"{device_bytes / 1024 ** 3:.1f}GB budget), so a decode pass never crosses "
                       f"the slow tier there would be anything to amortize; the draft's memory is "
                       f"better spent on resident weights. Pass speculative='on' to force it")

    recommendation = ratio = None
    if profile is not None:
        derivation = profile.derived.get("speculative_recommended")
        if derivation is not None:
            recommendation = derivation.value
            ratio = (derivation.inputs or {}).get("amortization_ratio")
    if recommendation is None:
        return False, ("this machine's amortization ratio was not measured, so there is no evidence "
                       "speculation would pay here; pass speculative='on' to force it")
    if recommendation:
        return True, (f"a streaming pass costs {ratio:.0f}x what device memory does on this "
                      f"machine, so one pass is worth several tokens")
    return False, (f"a streaming pass costs only {ratio:.0f}x what device memory does here, so the "
                   f"weights are not the bottleneck and the draft's memory is better spent on "
                   f"resident weights; pass speculative='on' to force it")


def lookahead_ceiling(profile=None, fallback=4):
    """How many tokens one verification pass on this machine is worth waiting for."""
    if profile is not None:
        derivation = profile.derived.get("speculative_lookahead")
        if derivation is not None and derivation.value:
            return max(1, int(derivation.value))
    return max(1, int(fallback))


def announce(enabled, reason, draft_name=None, lookahead=None):
    """Say once what was decided. A feature this consequential never turns on quietly."""
    if enabled:
        announce_once("spec-on",
                      f"speculative decoding is ON with draft {draft_name}: {reason}. Up to "
                      f"{lookahead} tokens are proposed per verification pass.", logging.INFO)
    else:
        announce_once("spec-off", f"speculative decoding is off: {reason}.", logging.INFO)
