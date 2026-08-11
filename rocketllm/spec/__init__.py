"""Speculative decoding.

The one optimization here that buys tokens rather than bytes. Everything else in this engine works
by moving fewer bytes or moving them from a faster tier; this works by getting more tokens out of a
streaming pass that was going to happen anyway, which is the third of the three ways an optimization
is allowed to pay for itself.

It is also the one that can be net-negative. The draft has to be resident, and what it occupies
comes out of the weight cache, so on a machine where the model already fits it costs residency to
save a pass that was cheap. The decision comes from the hardware profile's measured amortization
ratio and is never made silently.
"""
from .draft import (SPEC_AUTO, SPEC_CHOICES, SPEC_OFF, SPEC_ON, DraftIncompatible, DraftModel,
                    SamplingParams, SpeculationStats, SpeculativeDecoder, announce,
                    check_compatible, lookahead_ceiling, resolve_speculation, verify)

__all__ = [
    "DraftModel", "SpeculativeDecoder", "SpeculationStats", "SamplingParams",
    "DraftIncompatible", "check_compatible", "verify",
    "resolve_speculation", "lookahead_ceiling", "announce",
    "SPEC_AUTO", "SPEC_ON", "SPEC_OFF", "SPEC_CHOICES",
]
