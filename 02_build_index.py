"""
STEP 2 -- Build the search index (embed every document once).

Reads knowledge.jsonl, turns each document into a vector with the embedding
model, and saves the vectors + texts to disk. This is the "indexing" step --
you do it once, then reuse it for every question.

Run:  python 02_build_index.py
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from rag_utils import embed_texts

KNOWLEDGE_FILE = "knowledge.jsonl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 1. Load the documents.
texts = []
with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            texts.append(json.loads(line)["text"])

print(f"Loaded {len(texts)} documents. Embedding them...")

# 2. Embed them all into vectors (one row per document).
#embeddings = embed_texts(texts)

"""
Turn a list of strings into a numpy array of unit-length vectors.
Unit length means cosine similarity is just a dot product later.
"""
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
model = AutoModel.from_pretrained(EMBED_MODEL)
model.eval()

enc = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
with torch.no_grad():
    out = model(**enc)
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

print(f"Each document is now a vector of length {embeddings.shape[1]}.")

# 3. Save the index to disk so the next scripts can load it instantly.
np.save("rag_embeddings.npy", embeddings)
with open("rag_texts.json", "w", encoding="utf-8") as f:
    json.dump(texts, f, ensure_ascii=False, indent=2)

print("Saved index to rag_embeddings.npy + rag_texts.json")
print("Next: run  python 03_ask_rag.py")
