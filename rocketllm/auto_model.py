import importlib
from transformers import AutoConfig
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from rocketllm import RocketLlamaMlx

# Architectures that need a dedicated RocketLLM subclass because of a non-standard module layout
# (custom remote-code models). Everything else uses the generic RocketModel, which streams any
# standard *ForCausalLM (model.model.layers + lm_head / norm) and lets transformers own the
# forward pass, so newly released architectures work without code changes.
ARCH_OVERRIDES = {
    "ChatGLMModel": "RocketChatGLM",
    "ChatGLMForConditionalGeneration": "RocketChatGLM",
    "QWenLMHeadModel": "RocketQWen",
    "BaichuanForCausalLM": "RocketBaichuan",
    "BaiChuanForCausalLM": "RocketBaichuan",
    "InternLMForCausalLM": "RocketInternLM",
    "KimiK3ForConditionalGeneration": "RocketKimiK3",
}


class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if 'hf_token' in kwargs:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True,
                                                token=kwargs['hf_token'])
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        architectures = getattr(config, "architectures", None) or []
        arch = architectures[0] if architectures else ""

        cls_name = ARCH_OVERRIDES.get(arch)
        if cls_name is None:
            print(f"using generic RocketLLM streaming model for architecture: {arch or 'unknown'}")
            cls_name = "RocketModel"
        return "rocketllm", cls_name

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):

        if is_on_mac_os:
            return RocketLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        module, class_name = AutoModel.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
        module = importlib.import_module(module)
        class_ = getattr(module, class_name)
        return class_(pretrained_model_name_or_path, *inputs, **kwargs)
