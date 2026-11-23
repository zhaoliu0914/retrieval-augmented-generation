"""
A tiny vector database with a SWAPPABLE backend -- the heart of this kit.

  - If FAISS is installed  -> uses a real FAISS index (the production tool).
  - If FAISS is missing     -> falls back to pure numpy, same API, same results.

This mirrors how real systems work: application code talks to a clean interface
(add / search / save / load) and doesn't care which engine is underneath. That's
exactly how you'd later swap FAISS for Qdrant or Milvus without touching the rest.

Core concepts this teaches:
  * An index stores VECTORS only. You keep a side table mapping each vector to
    its text + metadata (the "id -> content" mapping every vector DB needs).
  * Vectors are unit-length, so inner product == cosine similarity.
  * The index persists to disk, so you build it once and reload instantly.
"""

import os
import json
import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class VectorDB:
    def __init__(self, dim):
        self.dim = dim
        self.texts = []
        self.metadatas = []
        self.backend = "faiss" if _HAS_FAISS else "numpy"
        if _HAS_FAISS:
            # IndexFlatIP = exact search by inner product (cosine, for unit vectors).
            # For millions of vectors you'd swap this for an approximate index like
            # faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT) -- far faster,
            # slightly less exact. Same .add()/.search() calls.
            self.index = faiss.IndexFlatIP(dim)
        else:
            self._vectors = np.zeros((0, dim), dtype="float32")

    def add(self, vectors, texts, metadatas=None):
        """Add a batch of vectors plus their text + optional metadata."""
        vectors = np.asarray(vectors, dtype="float32")
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if _HAS_FAISS:
            self.index.add(vectors)
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def search(self, query_vector, k=3):
        """Return the k most similar items as dicts: text, metadata, score."""
        q = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        if _HAS_FAISS:
            scores, idx = self.index.search(q, k)
            scores, idx = scores[0], idx[0]
        else:
            sims = self._vectors @ q[0]
            idx = sims.argsort()[::-1][:k]
            scores = sims[idx]
        results = []
        for i, s in zip(idx, scores):
            if i < 0:                      # FAISS pads with -1 if k > number stored
                continue
            results.append(
                {"text": self.texts[i], "metadata": self.metadatas[i], "score": float(s)}
            )
        return results

    def __len__(self):
        return len(self.texts)

    def save(self, folder):
        """Persist the index + the id->content side table to a folder."""
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "store.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"dim": self.dim, "backend": self.backend,
                 "texts": self.texts, "metadatas": self.metadatas},
                f, ensure_ascii=False, indent=2,
            )
        if _HAS_FAISS:
            faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        else:
            np.save(os.path.join(folder, "vectors.npy"), self._vectors)

    @classmethod
    def load(cls, folder):
        """Reload a saved database. No re-embedding of documents needed."""
        with open(os.path.join(folder, "store.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        db = cls(meta["dim"])
        db.texts = meta["texts"]
        db.metadatas = meta["metadatas"]
        if _HAS_FAISS:
            db.index = faiss.read_index(os.path.join(folder, "index.faiss"))
        else:
            db._vectors = np.load(os.path.join(folder, "vectors.npy"))
        return db
