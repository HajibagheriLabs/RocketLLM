"""Mixture-of-experts streaming.

A mixture is the case the whole engine is built for: the layer is enormous and the token reads
almost none of it. Everything here exists to find the experts without being told where they are, and
then to move only the ones a token actually routes to.

The rule the rest of the package relies on: there is NO cross-layer router lookahead. Layer L's
router runs inside layer L, so layer L+1's experts are unknowable until L has finished. Nothing in
here may be designed around one.
"""
from .detect import (LAYOUT_FUSED, LAYOUT_MODULE_LIST, ExpertContainer, ExpertLayout,
                     detect_expert_layout, resolve_top_k, summarize)
from .router import RouterSelection

__all__ = [
    "detect_expert_layout", "summarize", "resolve_top_k",
    "ExpertLayout", "ExpertContainer", "LAYOUT_MODULE_LIST", "LAYOUT_FUSED",
    "RouterSelection",
]
