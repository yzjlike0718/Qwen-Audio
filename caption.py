import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import os

# 如果您希望结果可复现，可以设置随机数种子。
torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

meld_dir = "/local/likh/Qwen2-Audio/data/meld"

for audio_name in os.listdir(meld_dir):
    audio_path = os.path.join(meld_dir, audio_name)

    query = tokenizer.from_list_format([
        {'audio': audio_path},
        {'text': 'what is that sound?'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)
