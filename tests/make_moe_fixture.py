"""Build a small mixture-of-experts checkpoint to run the correctness gate against.

The gate in test_streaming_gpu.py compares streamed generation against a full load, and it has to be
run on a mixture as well as a dense model -- expert streaming and whole-layer streaming are separate
paths and a change can break either one alone. There is no small public MoE that is convenient for
that: the ones that exist are tens of gigabytes, which is a long download to prove a code path works.

So this writes one. The weights are random, which is fine and in fact preferable: the gate asserts
that two runs of the *same* weights agree token for token, and random weights exercise the routing
just as well as trained ones while making the checkpoint small enough to rebuild on demand.

The architecture is Mixtral's, so what gets tested is a real expert layout rather than one invented
here. A tokenizer is copied in from an existing small model because RocketLLM loads one at startup;
any tokenizer will do, since the gate compares token ids and never the text they mean.

    python tests/make_moe_fixture.py --out ./moe-fixture
    python tests/test_streaming_gpu.py --model ./moe-fixture --compare

Nothing here needs an accelerator.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Small enough to build and load in seconds, big enough that every layer holds a real mixture and
#: the streamed path has several experts to choose between per token.
DEFAULTS = dict(hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=8,
                num_key_value_heads=4, num_local_experts=8, num_experts_per_tok=2,
                max_position_embeddings=512)


def skew_router(model, strength):
    """Bias every router toward a few experts, the way a trained one is biased.

    Random weights route almost perfectly uniformly, and uniform routing is the one case where
    hot-expert residency has nothing to exploit -- so a random fixture cannot tell whether the
    policy works, only that it does no harm. Damping the rows of each router's weight matrix by a
    Zipf-like profile concentrates the top-k on the first few experts and gives the residency policy
    something real to find. The skew is synthetic and says nothing about any particular model; what
    it exercises is the machinery that responds to skew.
    """
    scaled = 0
    for name, param in model.named_parameters():
        if param.ndim != 2 or not (name.endswith("gate.weight") or name.endswith("router.weight")):
            continue
        experts = param.shape[0]
        damping = torch.tensor([1.0 / (1.0 + index) ** strength for index in range(experts)],
                               dtype=param.dtype).unsqueeze(1)
        param.data.mul_(damping)
        scaled += 1
    return scaled


def build(out, tokenizer_id, seed=0, dtype=torch.float16, router_skew=0.0, **overrides):
    from transformers import AutoTokenizer, MixtralConfig, MixtralForCausalLM

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    settings = dict(DEFAULTS, **overrides)
    # Untied embeddings so the streamed sequence includes a separate lm_head shard, which is the
    # arrangement the engine takes its slower path for and therefore the one worth covering.
    config = MixtralConfig(vocab_size=len(tokenizer), tie_word_embeddings=False, **settings)

    torch.manual_seed(seed)
    model = MixtralForCausalLM(config).to(dtype).eval()
    skewed = skew_router(model, router_skew) if router_skew else 0
    model.config.torch_dtype = str(dtype).replace("torch.", "")
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    params = sum(p.numel() for p in model.parameters())
    note = f", router skew {router_skew} on {skewed} routers" if skewed else ""
    print(f"wrote {out} -- {params / 1e6:.0f}M parameters, "
          f"{settings['num_hidden_layers']} layers x {settings['num_local_experts']} experts, "
          f"top-k {settings['num_experts_per_tok']}, dtype {dtype}{note}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="./moe-fixture", help="directory to write the checkpoint to")
    parser.add_argument("--tokenizer", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="any tokenizer; the gate compares token ids, not text")
    parser.add_argument("--experts", type=int, default=DEFAULTS["num_local_experts"])
    parser.add_argument("--top-k", type=int, default=DEFAULTS["num_experts_per_tok"])
    parser.add_argument("--layers", type=int, default=DEFAULTS["num_hidden_layers"])
    parser.add_argument("--hidden", type=int, default=DEFAULTS["hidden_size"],
                        help="raise this to make a layer too large for a capped device budget")
    parser.add_argument("--intermediate", type=int, default=DEFAULTS["intermediate_size"])
    parser.add_argument("--heads", type=int, default=DEFAULTS["num_attention_heads"])
    parser.add_argument("--kv-heads", type=int, default=DEFAULTS["num_key_value_heads"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--router-skew", type=float, default=0.0,
                        help="Zipf exponent biasing each router toward its first few experts. "
                             "Random weights route uniformly, which is the one case hot-expert "
                             "residency cannot exploit; raise this to exercise the policy")
    args = parser.parse_args()

    build(args.out, args.tokenizer, seed=args.seed, dtype=getattr(torch, args.dtype),
          router_skew=args.router_skew,
          num_local_experts=args.experts, num_experts_per_tok=args.top_k,
          num_hidden_layers=args.layers, hidden_size=args.hidden,
          intermediate_size=args.intermediate, num_attention_heads=args.heads,
          num_key_value_heads=args.kv_heads)


if __name__ == "__main__":
    main()
