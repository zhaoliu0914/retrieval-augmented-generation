"""
STEP 3 -- Full RAG, with the vector database as the retrieval engine.

Same Retrieve -> Augment -> Generate flow as the RAG kit, but retrieval now goes
through the VectorDB (FAISS in production) instead of a hand-written numpy loop.
This is the production shape: a real index serving the LLM.

Run:  python 03_rag_with_db.py
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from embed import embed_texts
from vectordb import VectorDB

LLM_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
DB_FOLDER = "./nimbus_db"
TOP_K = 3

# Load the vector DB + the generator LLM.
db = VectorDB.load(DB_FOLDER)
tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
model = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=torch.float32)
model.eval()

question = "How many strawberries can the HarvestHop R9 pick per hour?"

# 1. RETRIEVE (via the vector database)
q_vec = embed_texts([question])[0]
hits = db.search(q_vec, k=TOP_K)
print(f"\n=== Retrieved from vector DB (backend: {db.backend.upper()}) ===")
for h in hits:
    print(f"  [{h['score']:.3f}] {h['text']}")

# 2. AUGMENT
context = "\n".join(f"- {h['text']}" for h in hits)
system_prompt = (
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say you don't know.\n\n"
    f"Context:\n{context}"
)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question},
]

# 3. GENERATE
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=120, do_sample=False)

reply = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("\n=== ANSWER ===")
print("Q:", question)
print("A:", reply)
