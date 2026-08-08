from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from .llama_mlx import RocketLlamaMlx
    from .auto_model import AutoModel
else:
    # Core entry points. These have no model-specific optional dependencies, so a plain
    # `import rocketllm` always works as long as torch/transformers are installed.
    from .base import RocketModel
    from .auto_model import AutoModel
    from .utils import split_and_save_layers
    from .utils import NotEnoughSpaceException

    # Dedicated subclasses for a handful of custom-architecture models. Some of them pull in
    # optional extras (e.g. the Baichuan tokenizer needs `sentencepiece`). Import them defensively
    # so a missing optional dependency for one niche family never breaks the whole package; the
    # generic RocketModel path keeps working regardless.
    import warnings as _warnings

    for _name, _module in (
        ("RocketLlama", ".llama"),
        ("RocketChatGLM", ".chatglm"),
        ("RocketQWen", ".qwen"),
        ("RocketQWen2", ".qwen2"),
        ("RocketBaichuan", ".baichuan"),
        ("RocketInternLM", ".internlm"),
        ("RocketMistral", ".mistral"),
        ("RocketMixtral", ".mixtral"),
        ("RocketKimiK3", ".kimi_k3"),
    ):
        try:
            _mod = __import__(__name__ + _module, fromlist=[_name])
            globals()[_name] = getattr(_mod, _name)
        except Exception as _e:  # noqa: BLE001 - optional family, keep package importable
            _warnings.warn(
                f"rocketllm: optional model class {_name} is unavailable ({_e}). "
                f"This only affects that specific model family; the generic streaming path still works."
            )

