"""
Shared RAG helpers: turn text into vectors (embeddings), and find the most
relevant documents for a query. Imported by the numbered step scripts.

The embedding model is a small, standard one (all-MiniLM-L6-v2, ~90 MB, 384-dim).
We load it with plain transformers -- no sentence-transformers needed -- so the
whole mechanism is visible.
"""

import os

# Offline mode: use locally cached models, never phone home. Essential in China
# without a VPN. Must be set BEFORE importing transformers.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_tokenizer = None
_model = None


def _load_embedder():
    """Load the embedding model once and reuse it."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        _model = AutoModel.from_pretrained(EMBED_MODEL)
        _model.eval()


def _mean_pool(last_hidden_state, attention_mask):
    """
    Average the token vectors into one vector per text.
    We ignore padding tokens by using the attention mask as weights.
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_texts(texts):
    """
    Turn a list of strings into a numpy array of unit-length vectors.
    Unit length means cosine similarity is just a dot product later.
    """
    _load_embedder()
    enc = _tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        out = _model(**enc)
    emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # make each vector length 1
    return emb.numpy().astype("float32")


def retrieve(query, doc_embeddings, doc_texts, k=3):
    """
    Embed the query, score it against every document by cosine similarity,
    and return the top-k (text, score) pairs, most relevant first.
    """
    q = embed_texts([query])[0]  # shape: (384,)
    scores = doc_embeddings @ q  # dot product == cosine (vectors are unit length)
    top_idx = scores.argsort()[::-1][:k]  # indices of the highest scores
    return [(doc_texts[i], float(scores[i])) for i in top_idx]
