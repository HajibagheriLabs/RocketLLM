"""RocketLLM: run language models far larger than the device they are running on.

Importing this package must work on a base install, on every platform, with none of the optional
extras present. Everything below that could need an optional package is imported defensively and
degrades to a warning naming what is missing -- the generic streaming path never depends on any of
them, so a missing extra costs one model family or one backend, never the package.
"""
import warnings as _warnings
from sys import platform

# Core entry points. These have no optional dependencies beyond torch/transformers, so a plain
# `import rocketllm` always works once the base install is in place -- on every platform.
from .base import RocketModel
from .auto_model import AutoModel
from .utils import split_and_save_layers
from .utils import NotEnoughSpaceException

is_on_mac_os = (platform == "darwin")

# Dedicated subclasses for a handful of custom-architecture models, plus the Apple Silicon MLX
# backend. Some pull in optional extras (the Baichuan tokenizer needs `sentencepiece`; the MLX path
# needs `mlx` and `psutil`), so each is imported defensively: a missing optional dependency for one
# niche family or one backend must never break the whole package.
_OPTIONAL_CLASSES = [
    ("RocketLlama", ".llama"),
    ("RocketChatGLM", ".chatglm"),
    ("RocketQWen", ".qwen"),
    ("RocketQWen2", ".qwen2"),
    ("RocketBaichuan", ".baichuan"),
    ("RocketInternLM", ".internlm"),
    ("RocketMistral", ".mistral"),
    ("RocketMixtral", ".mixtral"),
    ("RocketKimiK3", ".kimi_k3"),
]

if is_on_mac_os:
    # Only attempted on Apple Silicon, since that is the only place MLX runs. It used to be an
    # unguarded import here, which meant `import rocketllm` on a Mac without the mlx extra ended in
    # an ImportError traceback -- on the one platform where that extra is genuinely optional.
    _OPTIONAL_CLASSES.append(("RocketLlamaMlx", ".llama_mlx"))

for _name, _module in _OPTIONAL_CLASSES:
    try:
        _mod = __import__(__name__ + _module, fromlist=[_name])
        globals()[_name] = getattr(_mod, _name)
    except Exception as _e:  # noqa: BLE001 - optional family, keep package importable
        _warnings.warn(
            f"rocketllm: optional model class {_name} is unavailable ({_e}). "
            f"This only affects that specific model family; the generic streaming path still works."
        )
