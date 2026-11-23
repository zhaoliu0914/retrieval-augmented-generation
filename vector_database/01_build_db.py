"""
STEP 1 -- Build and persist the vector database.

Embeds every document, adds the vectors (plus text + metadata) to the VectorDB,
and saves it to ./nimbus_db on disk. This is the production "ingestion" step:
you run it once (or when documents change), not on every query.

Run:  python 01_build_db.py
"""

import json
from embed import embed_texts, EMBED_DIM
from vectordb import VectorDB

KNOWLEDGE_FILE = "knowledge.jsonl"
DB_FOLDER = "./nimbus_db"

# 1. Load documents + their metadata.
texts, metadatas = [], []
with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        texts.append(rec["text"])
        metadatas.append({"id": rec["id"], "source": rec.get("source", "")})

# 2. Embed them.
print(f"Embedding {len(texts)} documents...")
vectors = embed_texts(texts)

# 3. Put them in the vector database.
db = VectorDB(dim=EMBED_DIM)
db.add(vectors, texts, metadatas)

# 4. Persist to disk.
db.save(DB_FOLDER)

print(f"Backend in use: {db.backend.upper()}")      # FAISS or NUMPY (fallback)
print(f"Stored {len(db)} vectors. Saved to {DB_FOLDER}/")
print("Next: python 02_search.py")
