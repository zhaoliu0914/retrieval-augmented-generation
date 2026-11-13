"""
STEP 3 -- RAG: Retrieve, Augment, Generate.

Same fictional question as step 1, but now:
  1. RETRIEVE: find the most relevant documents from the index.
  2. AUGMENT: paste them into the prompt as context.
  3. GENERATE: let the model answer using that context.

The model should now answer correctly -- not because it learned the fact, but
because we handed it the right page to read.

Run:  python 03_ask_rag.py
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from rag_utils import retrieve

TOP_K = 3
LLM_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Load the index built in step 2 ---
doc_embeddings = np.load("rag_embeddings.npy")
with open("rag_texts.json", "r", encoding="utf-8") as f:
    doc_texts = json.load(f)

# --- Load the generator LLM ---
tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
model = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=torch.float32)
model.eval()

question = "How many strawberries can the HarvestHop R9 pick per hour?"

# --- 1. RETRIEVE ---
#hits = retrieve(question, doc_embeddings, doc_texts, k=TOP_K)
#q = embed_texts([query])[0]            # shape: (384,)

# embed_texts
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(EMBED_MODEL)
embed_model.eval()

enc = embed_tokenizer(question, padding=True, truncation=True, max_length=256, return_tensors="pt")
with torch.no_grad():
    out = embed_model(**enc)
#emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])

"""
Average the token vectors into one vector per text.
We ignore padding tokens by using the attention mask as weights.
"""
mask = enc["attention_mask"].unsqueeze(-1).float()
summed = (out.last_hidden_state * mask).sum(dim=1)
counts = mask.sum(dim=1).clamp(min=1e-9)
emb = summed / counts
emb = torch.nn.functional.normalize(emb, p=2, dim=1)   # make each vector length 1
embeddings = emb.numpy().astype("float32")

# retrieve
q = embeddings[0]
scores = doc_embeddings @ q            # dot product == cosine (vectors are unit length)
top_idx = scores.argsort()[::-1][:TOP_K]   # indices of the highest scores
hits = [(doc_texts[i], float(scores[i])) for i in top_idx]

print("\n=== Retrieved context (most relevant first) ===")
for text, score in hits:
    print(f"  [{score:.3f}] {text}")

# --- 2. AUGMENT: build a prompt that includes the retrieved context ---
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

# --- 3. GENERATE ---
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
        )

reply = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("\n=== RAG ANSWER ===")
print("Q:", question)
print("A:", reply)
print("\n(Now it should say ~1,200 per hour -- straight from the retrieved document.)")
