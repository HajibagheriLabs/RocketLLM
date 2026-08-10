"""What fits where, measured continuously rather than modelled.

Everything in this package answers a sizing question about the machine as it is *right now*, not as
it was at load. Free device memory moves under a streaming run -- the KV cache grows with every
token, activations come and go, another process may take a slice of the card -- so a budget computed
once at startup is wrong by the end of the first generation.
"""
from .budget import AllocatorSetup, BudgetSample, VramBudget, configure_allocator

__all__ = ["VramBudget", "BudgetSample", "AllocatorSetup", "configure_allocator"]
