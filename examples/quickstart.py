"""Smallest possible RocketLLM run: stream a 70B checkpoint through a small GPU.

The model is never fully resident. AutoModel picks the right class from the checkpoint's
architecture, splits it into per-layer shards on first use, and streams those shards in as
each layer runs.
"""

from rocketllm import AutoModel

MAX_LENGTH = 128

# A Hugging Face repo id, or a local path to an already-downloaded checkpoint.
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-70B-instruct")

input_text = [
    'What is the capital of United States?',
]

input_tokens = model.tokenizer(input_text,
                               return_tensors="pt",
                               return_attention_mask=False,
                               truncation=True,
                               max_length=MAX_LENGTH,
                               padding=True)

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True)

print(model.tokenizer.decode(generation_output.sequences[0]))
