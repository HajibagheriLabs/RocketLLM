"""Architecture -> streaming-class mapping.

These run offline. The mapping is driven entirely by ``config.architectures[0]``, so a temporary
directory holding a one-line config exercises the real ``get_module_class`` path -- config load,
architecture extraction, override lookup, generic fallback -- without fetching a single checkpoint.
"""
import json
import tempfile
import unittest
from pathlib import Path

from rocketllm.auto_model import ARCH_OVERRIDES, AutoModel

# Spelled out rather than derived from ARCH_OVERRIDES: comparing the table against itself would
# pass no matter how it was edited. These are the architectures whose module layout the generic
# streaming path cannot handle, and the class each one must reach.
EXPECTED_OVERRIDES = {
    "ChatGLMModel": "RocketChatGLM",
    "ChatGLMForConditionalGeneration": "RocketChatGLM",
    "QWenLMHeadModel": "RocketQWen",
    "BaichuanForCausalLM": "RocketBaichuan",
    "BaiChuanForCausalLM": "RocketBaichuan",
    "InternLMForCausalLM": "RocketInternLM",
    "KimiK3ForConditionalGeneration": "RocketKimiK3",
}


def module_class_for(architecture):
    """Resolve the streaming class for a checkpoint announcing ``architecture``.

    ``model_type`` stays "llama" so AutoConfig resolves against a type transformers always knows
    and never reaches the network; the code under test only ever reads ``architectures[0]``.
    """
    config = {"model_type": "llama"}
    if architecture is not None:
        config["architectures"] = [architecture]
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return AutoModel.get_module_class(tmp)


class TestArchOverrides(unittest.TestCase):
    def test_override_table_matches_the_supported_custom_architectures(self):
        self.assertEqual(ARCH_OVERRIDES, EXPECTED_OVERRIDES)

    def test_custom_architectures_resolve_to_their_dedicated_class(self):
        for architecture, expected in EXPECTED_OVERRIDES.items():
            with self.subTest(architecture=architecture):
                self.assertEqual(module_class_for(architecture), ("rocketllm", expected))

    def test_override_targets_are_exported_by_the_package(self):
        """A table entry naming a class that does not exist would only fail at load time."""
        import rocketllm

        for expected in sorted(set(EXPECTED_OVERRIDES.values())):
            with self.subTest(cls=expected):
                self.assertTrue(hasattr(rocketllm, expected),
                                f"ARCH_OVERRIDES names {expected}, which the package does not export")


class TestGenericFallback(unittest.TestCase):
    """Standard architectures deliberately have no entry.

    RocketModel streams any ordinary ``*ForCausalLM`` and lets transformers own the forward pass,
    which is what lets a newly released architecture work with no change here. An architecture
    appearing in ARCH_OVERRIDES is the exception, not the rule.
    """

    def test_standard_architectures_use_the_generic_streaming_class(self):
        for architecture in ("LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM",
                             "Qwen2ForCausalLM", "GemmaForCausalLM", "Phi3ForCausalLM"):
            with self.subTest(architecture=architecture):
                self.assertEqual(module_class_for(architecture), ("rocketllm", "RocketModel"))

    def test_an_unreleased_architecture_needs_no_code_change(self):
        self.assertEqual(module_class_for("SomeFutureModelForCausalLM"),
                         ("rocketllm", "RocketModel"))

    def test_a_config_without_architectures_falls_back_to_the_generic_class(self):
        self.assertEqual(module_class_for(None), ("rocketllm", "RocketModel"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
