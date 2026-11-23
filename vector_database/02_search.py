"""
STEP 2 -- Pure search against the saved database (no LLM yet).

Loads the database from disk (note: the documents are NOT re-embedded -- the
index is already built), embeds just the query, and returns the closest matches
with their similarity scores and metadata. This is the "retrieval" half of RAG,
now served by the vector database.

Run:  python 02_search.py
"""

from embed import embed_texts
from vectordb import VectorDB

DB_FOLDER = "./nimbus_db"

db = VectorDB.load(DB_FOLDER)
print(f"Loaded {len(db)} vectors (backend: {db.backend.upper()}).\n")

queries = [
    "How fast can the strawberry robot pick fruit?",
    "Where is the company's research lab?",
    "Who runs the company?",
]

for query in queries:
    q_vec = embed_texts([query])[0]
    hits = db.search(q_vec, k=2)
    print(f"Q: {query}")
    for h in hits:
        print(f"   [{h['score']:.3f}] ({h['metadata']['source']}) {h['text']}")
    print()
