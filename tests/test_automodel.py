import unittest

from rocketllm.auto_model import AutoModel



class TestAutoModel(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_auto_model_should_return_correct_model(self):
        mapping_dict = {
            'garage-bAInd/Platypus2-7B': 'RocketLlama',
            'Qwen/Qwen-7B': 'RocketQWen',
            'internlm/internlm-chat-7b': 'RocketInternLM',
            'THUDM/chatglm3-6b-base': 'RocketChatGLM',
            'baichuan-inc/Baichuan2-7B-Base': 'RocketBaichuan',
            'mistralai/Mistral-7B-Instruct-v0.1': 'RocketMistral',
            'mistralai/Mixtral-8x7B-v0.1': 'RocketMixtral'
        }


        for k,v in mapping_dict.items():
            module, cls = AutoModel.get_module_class(k)
            self.assertEqual(cls, v, f"expecting {v}")

