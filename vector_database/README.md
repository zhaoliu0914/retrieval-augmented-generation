# Learning vector databases — the production RAG engine

In the RAG kit you retrieved documents with a hand-written numpy loop. That's
perfect for 10 documents and fine for a few thousand — but it compares the query
against *every* vector one by one, so it gets slow at scale. Production systems
replace that loop with a **vector database**: a specialized index that searches
millions of vectors in milliseconds, persists to disk, and supports add/update/
delete and metadata.

This kit upgrades your RAG retrieval to **FAISS** (Meta's standard vector-search
library), wrapped so it still runs on your Mac even if FAISS won't install.

---

## What's in this folder

| File | What it is |
|------|-----------|
| `requirements-vectordb.txt` | Adds `faiss-cpu` to the RAG stack (optional) |
| `knowledge.jsonl`           | The same 10 fictional docs, now with metadata |
| `embed.py`                  | Embedding helper (text → vectors) |
| `vectordb.py`               | **The star:** a VectorDB with a FAISS/numpy swappable backend |
| `01_build_db.py`            | Step 1 — embed + store + persist the database |
| `02_search.py`              | Step 2 — load and search it (no LLM) |
| `03_rag_with_db.py`         | Step 3 — full RAG using the DB as the engine |

---

## Setup (reuse your RAG environment)

Easiest path — the RAG venv already has everything except FAISS:

```bash
cd ~/Projects/rag && source .venv/bin/activate   # your existing RAG env
# copy this kit's files into a folder, then:
pip install faiss-cpu==1.8.0.post1
```

**If that install fails on Big Sur** (Intel Mac can't always match the macOS
version FAISS targets): don't worry at all. `vectordb.py` automatically falls
back to a numpy backend with the identical API. The kit runs fine and teaches
every concept — you just won't get FAISS's speed-at-scale until you run it on
your Windows/WSL2 box (where `pip install faiss-cpu` works cleanly). You'll know
which backend is active because every script prints `FAISS` or `NUMPY`.

The embedding model (all-MiniLM-L6-v2) is already cached from the RAG kit, so no
new download and no VPN needed.

---

## The three steps

```bash
python 01_build_db.py    # embed 10 docs, store vectors + metadata, save to ./nimbus_db
python 02_search.py      # load the saved DB, run some searches, see scores + metadata
python 03_rag_with_db.py # full RAG: retrieve via the DB, then generate an answer
```

Notice in step 2 that the documents are **not re-embedded** — the index was
built once in step 1 and reloaded from disk. That persistence is a core reason
vector databases exist.

---

## What a vector database adds over the numpy loop

| | numpy loop (RAG kit) | vector database (this kit) |
|---|---|---|
| Search method | compare to every vector | smart index (exact or approximate) |
| Speed at 1M+ vectors | slow (linear scan) | milliseconds |
| Persistence | you save arrays yourself | built in (`save`/`load`) |
| Add / delete | rebuild the array | incremental |
| Metadata | manual | stored alongside vectors |

### Exact vs approximate search (the key scaling idea)

- **Exact** (`IndexFlatIP`, what this kit uses): checks every vector, always
  returns the true nearest neighbors. Perfect accuracy, but linear time.
- **Approximate (ANN)** (`IndexHNSWFlat`, `IndexIVFFlat`): clever index
  structures that find the nearest neighbors *almost* always, in sub-linear
  time. This is what makes million-vector search fast. You trade a tiny bit of
  recall for a huge speedup. See the comment in `vectordb.py` for the one-line
  swap to HNSW.

### The id → content mapping

A vector index stores **only vectors** (and integer ids). It does *not* store
your text. So you keep a side table mapping each vector's position to its text +
metadata — that's the `texts` / `metadatas` lists in `vectordb.py`. Every vector
database works this way under the hood; some (Qdrant, Milvus) manage the side
table for you.

---

## The production landscape

What you'd reach for beyond this learning kit:

- **FAISS** — a *library*, runs inside your Python process. Fast, battle-tested,
  no server to run. Great default; what you're using here. You manage metadata
  yourself.
- **Qdrant / Milvus / Weaviate** — vector *database servers*. You run them (often
  via Docker) and talk to them over the network. They add metadata filtering
  ("only search documents where source = 'statutes' and year > 2020"),
  horizontal scaling, and concurrent access. This is where you'd go for a serious
  multi-user legal system.
- **pgvector** — a PostgreSQL extension. If your data already lives in Postgres,
  it adds vector search right next to your normal tables. Popular for teams who
  want one database instead of two.

The good news: they all share the shape you just learned — add vectors, search
top-k, map ids to content. Moving from FAISS to Qdrant is an API change, not a
new concept. That's exactly why `vectordb.py` hides the backend behind a clean
interface.

---

## Now make it yours

1. **Scale it up.** Duplicate the lines in `knowledge.jsonl` into hundreds of
   docs and watch build/search still feel instant — then imagine millions, where
   the numpy loop would crawl but an ANN index stays fast.
2. **Try approximate search.** In `vectordb.py`, swap `IndexFlatIP` for
   `faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)` and compare results.
3. **Use the metadata.** Each hit carries its `source`. Add a filter step after
   `search()` that keeps only hits from a chosen source — a preview of the
   metadata filtering real vector DBs do natively.

---

## Where this fits your law system

This is the retrieval engine for the architecture we discussed:

- **Fine-tuning** → the model's legal voice, format, and reasoning style.
- **RAG** → pulls the authoritative legal text into the prompt (with citations).
- **Vector database** → makes that retrieval fast and scalable over a huge corpus
  of statutes and cases, with metadata filtering (jurisdiction, year, court).

You've now built all three pieces at learning scale. A production legal system is
the same shapes, bigger: a real vector DB server, a strong generator model,
careful chunking, and a human in the loop for anything consequential.
