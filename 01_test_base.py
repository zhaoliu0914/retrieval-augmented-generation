"""
STEP 1 -- Ask the model WITHOUT retrieval (no RAG yet).

The question is about Nimbus Robotics, a FICTIONAL company that exists only in
knowledge.jsonl. The model was never trained on it, so here it will either admit
it doesn't know or -- more likely -- hallucinate a confident wrong answer.
Remember this: it's the "before" picture.

Run:  python 01_test_base.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
model = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=torch.float32)
model.eval()

question = "How many strawberries can the HarvestHop R9 pick per hour?"
messages = [{"role": "user", "content": question}]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

reply = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("\n=== NO RAG (model answers from memory only) ===")
print("Q:", question)
print("A:", reply)
print("\n(The model can't know this fictional fact -- expect 'I don't know' or a made-up answer.)")
