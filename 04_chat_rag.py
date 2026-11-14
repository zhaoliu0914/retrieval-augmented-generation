"""
STEP 4 (optional) -- Chat with your RAG system in a loop.

Type any question about Nimbus Robotics; it retrieves, augments, and answers.
It also prints which document it retrieved, so you can SEE retrieval working.
Type 'quit' to exit.

Try a question whose answer IS in knowledge.jsonl (e.g. "Who is the CEO?") and
one that is NOT (e.g. "What color is the logo?") -- watch it say it doesn't know
rather than making something up.

Run:  python 04_chat_rag.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag_utils import retrieve

LLM_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
TOP_K = 3

doc_embeddings = np.load("rag_embeddings.npy")
with open("rag_texts.json", "r", encoding="utf-8") as f:
    doc_texts = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
model = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=torch.float32)
model.eval()

print("RAG chat ready. Ask about Nimbus Robotics. Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in {"quit", "exit", "q"}:
        break
    if not question:
        continue

    hits = retrieve(question, doc_embeddings, doc_texts, k=TOP_K)
    print(f"  (retrieved: {hits[0][0]})")

    context = "\n".join(f"- {text}" for text, _ in hits)
    system_prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    reply = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"Model: {reply}\n")
