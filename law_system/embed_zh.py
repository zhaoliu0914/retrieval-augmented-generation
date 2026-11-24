"""
Chinese text embedder for legal case retrieval.

Uses BAAI/bge-small-zh-v1.5 -- a small, strong Chinese embedding model that runs
on CPU. Two important details specific to bge models:
  * bge uses CLS pooling (the first token), NOT mean pooling.
  * For retrieval, SHORT QUERIES get a special instruction prefix; the stored
    documents do NOT. This asymmetry improves retrieval quality.

EMBED_MODEL can be a Hugging Face id (if downloaded via HF) OR a local folder
path (if you downloaded via ModelScope in China -- snapshot_download returns one).
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"   # or a local path from ModelScope
EMBED_DIM = 512
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："  # bge-zh retrieval prefix

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        _model = AutoModel.from_pretrained(EMBED_MODEL)
        _model.eval()


def embed_texts(texts, is_query=False, batch_size=64):
    """
    Turn a list of Chinese strings into unit-length vectors.
    Set is_query=True for the user's search text (adds the bge instruction prefix).
    Batched so a large corpus doesn't blow up memory.
    """
    _load()
    if is_query:
        texts = [QUERY_INSTRUCTION + t for t in texts]

    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = _tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            out = _model(**enc)
        emb = out.last_hidden_state[:, 0]          # CLS pooling (bge uses the [CLS] token)
        emb = F.normalize(emb, p=2, dim=1)         # unit length -> cosine == dot product
        all_emb.append(emb.numpy())
        print(f"    embedded {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.vstack(all_emb).astype("float32")
