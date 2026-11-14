import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    num = tokenizer("Hello Model!!!!!")
    print(num["input_ids"])
    print(tokenizer.decode(num["input_ids"]))


def tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    model = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=torch.float32)

    inputs = tokenizer("The capital of Italy is Roma", return_tensors="pt")
    print(f"inputs = {inputs}")

    output_nums = model.generate(**inputs, max_new_tokens=20)
    print(f"output_nums = {output_nums}")

    output_text = tokenizer.decode(output_nums[0], skip_special_tokens=True)
    print(f"output_text = {output_text}")


if __name__ == '__main__':
    # tokenizer()
    tokenizer_model()
