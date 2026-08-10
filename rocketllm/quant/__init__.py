"""Pre-quantized checkpoint intake.

RocketLLM imports checkpoints that were already quantized by someone else; it never quantizes a
model itself. This package is the one place that knows how the resulting formats lay a weight out
on disk, and it presents all of them through a single object -- :class:`PackedWeight` -- so the
streaming path and the cache can size and move weights without knowing which format they came from.

The readers for individual formats (compressed-tensors, bitsandbytes) are optional. Importing this
package never requires them: a format's reader is imported at the point it is needed, so a machine
that has none of them still loads every checkpoint the others cover.
"""
from .registry import (COMPANION_SUFFIXES, PackedWeight, QuantBackend, TensorSpec, decision_table,
                       detect_backend, quant_method_of, register_backend, registered_backends)

__all__ = ["PackedWeight", "QuantBackend", "TensorSpec", "COMPANION_SUFFIXES", "decision_table",
           "detect_backend", "quant_method_of", "register_backend", "registered_backends"]
