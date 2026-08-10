"""Reading a router's choice as it is made.

A fused expert tensor has no per-expert module to hang a hook on, so the only way to know which rows
a token needs is to watch the router decide. The top-k is therefore fetched WITHIN a layer, once
that layer's router has fired. There is no cross-layer lookahead and this class does not imply one:
it observes layer L's router while layer L is running, which is the only moment the information
exists.

What counts as a usable router output is deliberately narrow: exactly one floating-point tensor
whose last dimension is the expert count. That is a score per expert, and taking its top-k
reproduces the model's own selection, because every routing rule in use picks the k highest scores
and top-k is invariant under the softmax or sigmoid that may sit between the two.

Everything else -- a router returning indices it already computed, a compound gating module
returning five tensors, a rule this code has never seen -- reads as *unknown*, and unknown means the
container streams every row. That is the whole-layer behaviour the engine had before, so it is
correct and merely slow. The alternative, guessing at an unfamiliar routing rule, streams the wrong
experts and produces confidently wrong tokens, and no amount of bandwidth is worth that.
"""
import logging

import torch

log = logging.getLogger(__name__)

#: How deep to look inside a router's return value for the score tensor. Routers return a tensor, a
#: tuple, or a ModelOutput; none of them nest meaningfully deeper than this.
_MAX_DEPTH = 3


def score_tensors(output, num_experts, depth=0):
    """Floating-point tensors in a router's output that carry one score per expert."""
    if depth > _MAX_DEPTH:
        return []
    if torch.is_tensor(output):
        if (output.is_floating_point() and output.ndim >= 1
                and int(output.shape[-1]) == num_experts):
            return [output]
        return []
    if isinstance(output, dict):
        children = list(output.values())
    elif isinstance(output, (list, tuple)):
        children = list(output)
    else:
        return []
    found = []
    for child in children:
        found.extend(score_tensors(child, num_experts, depth + 1))
    return found


class RouterSelection:
    """The experts one fused container's router chose, for the tokens now in flight.

    The value is consumed by whoever reads it: :meth:`take` clears the slot. That makes the failure
    mode safe by construction -- if the router hook does not fire for any reason, the container's
    pre-hook sees "unknown" rather than the previous token's answer, and reads every row.
    """

    __slots__ = ("num_experts", "top_k", "path", "_selected", "_unusable_logged", "observations")

    def __init__(self, num_experts, top_k, path=""):
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.path = path
        self._selected = None
        self._unusable_logged = False
        self.observations = 0

    def observe(self, output):
        """Record the selection implied by a router's output. Returns what was recorded."""
        candidates = score_tensors(output, self.num_experts)
        if len(candidates) != 1:
            # Zero: the router does not hand back per-expert scores. More than one: the structure
            # is not saying which of them drives the experts. Either way, do not guess.
            self._selected = None
            if not self._unusable_logged:
                self._unusable_logged = True
                log.info("%s: the router returns %d per-expert score tensors, so its selection "
                         "cannot be read; this container streams all %d experts",
                         self.path or "fused experts", len(candidates), self.num_experts)
            return None

        scores = candidates[0].detach()
        k = min(self.top_k, self.num_experts)
        # float() so the comparison is well-defined for every dtype a router may run in; the tensor
        # is one row per token and costs nothing to widen.
        chosen = torch.topk(scores.reshape(-1, self.num_experts).float(), k, dim=-1).indices
        # Crossing to the host is unavoidable and is the cost of routing: the rows to read are a
        # data-dependent decision and the read is a host-side file operation. It is also why there
        # is no lookahead past this layer.
        self._selected = tuple(sorted(set(chosen.reshape(-1).tolist())))
        self.observations += 1
        return self._selected

    def take(self):
        """The recorded selection, clearing it. ``None`` means "unknown, read everything"."""
        selected, self._selected = self._selected, None
        return selected

    def reset(self):
        self._selected = None

    def __repr__(self):
        return (f"<RouterSelection {self.path} {self.num_experts} experts top-k {self.top_k} "
                f"observed {self.observations}>")
