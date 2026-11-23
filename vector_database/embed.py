"""
Embedding helper (same approach as the RAG kit). Turns text into unit-length
vectors with the small all-MiniLM-L6-v2 model, using plain transformers.
Self-contained so this kit stands alone.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # use cached models; no network
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # essential in China without VPN

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        _model = AutoModel.from_pretrained(EMBED_MODEL)
        _model.eval()


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_texts(texts):
    _load()
    enc = _tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        out = _model(**enc)
    emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.numpy().astype("float32")
