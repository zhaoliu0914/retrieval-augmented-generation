import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LLM_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)


if __name__ == '__main__':
    tokenizer()