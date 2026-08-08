
from transformers import GenerationConfig

from .base import RocketModel



class RocketMistral(RocketModel):


    def __init__(self, *args, **kwargs):


        super(RocketMistral, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False
    def get_generation_config(self):
        return GenerationConfig()


