# Learning RAG on your Mac — a hands-on kit

Retrieval-Augmented Generation (RAG) is how you give a model **new facts** it was
never trained on — the thing fine-tuning couldn't reliably do. Instead of
changing the model's weights, you *retrieve* the relevant text at question time
and paste it into the prompt, so the model reads the answer instead of recalling it.

This kit teaches RAG the same before→after way the fine-tuning kit did, using a
small **fictional** knowledge base (so when the model answers correctly, you
*know* the fact came from retrieval, not from pretraining).

Runs on your Intel Mac (macOS 11, torch 2.2.2), CPU-only, fully offline once set up.

---

## What's in this folder

| File | What it is |
|------|-----------|
| `requirements-rag.txt`   | The package stack (same safe set as fine-tuning, minus the training libs) |
| `knowledge.jsonl`        | 10 fictional documents about "Nimbus Robotics" — the facts to retrieve |
| `rag_utils.py`           | Shared helpers: `embed_texts()` and `retrieve()` |
| `01_test_base.py`        | Step 1 — ask WITHOUT retrieval (model can't know) |
| `02_build_index.py`      | Step 2 — embed the documents into a searchable index |
| `03_ask_rag.py`          | Step 3 — retrieve + augment + generate (now it answers) |
| `04_chat_rag.py`         | Step 4 (optional) — chat with your RAG system |

---

## One-time setup (a NEW, separate environment)

This keeps your fine-tuning setup completely untouched. Your Hugging Face model
cache is shared across environments, so **SmolLM2-360M is already available** —
the only new download is the small embedding model.

```bash
# 1. New folder + new virtual environment
mkdir -p ~/Projects/rag && cd ~/Projects/rag
python3 -m venv .venv && source .venv/bin/activate

# 2. Put all the kit files in this folder, then install
pip install torch==2.2.2
pip install -r requirements-rag.txt

# 3. Download the embedding model ONCE (~90 MB).
#    In China: turn your VPN ON for just this step, then you can turn it off.
hf download sentence-transformers/all-MiniLM-L6-v2
```

After that download, everything runs **offline** — the scripts set
`HF_HUB_OFFLINE=1` themselves, so no VPN is needed to run them.

---

## The four steps

Run in order, from inside the folder.

### Step 1 — Ask without RAG

```bash
python 01_test_base.py
```

Asks "How many strawberries can the HarvestHop R9 pick per hour?" — a fact that
exists only in `knowledge.jsonl`. The model can't know it, so it says "I don't
know" or **hallucinates** a confident wrong number. This is the problem RAG solves.

### Step 2 — Build the index

```bash
python 02_build_index.py
```

Turns each of the 10 documents into a vector (a list of 384 numbers that captures
its meaning) and saves them to `rag_embeddings.npy` + `rag_texts.json`. You do
this once; adding/removing documents means re-running this step.

### Step 3 — Ask with RAG

```bash
python 03_ask_rag.py
```

Same question, but now it **R**etrieves the most relevant documents, **A**ugments
the prompt with them, and **G**enerates. It prints what it retrieved (with
similarity scores) and then answers correctly — ~1,200 per hour — straight from
the document. That's RAG.

### Step 4 — Chat (optional)

```bash
python 04_chat_rag.py
```

Ask anything about Nimbus Robotics. Try one answerable question ("Who is the
CEO?") and one that isn't in the data ("What color is the logo?") — watch it
correctly say it doesn't know instead of inventing an answer. That refusal is
RAG working *well*.

---

## How RAG works (the mental model)

```
        INDEXING (once)                    QUERY TIME (every question)
   ┌────────────────────────┐        ┌──────────────────────────────────┐
   documents → embed → vectors       question → embed → vector
                                                        │
                                              compare to all doc vectors
                                              (cosine similarity)
                                                        │
                                              take top-k closest documents
                                                        │
                                     paste them into the prompt as "context"
                                                        │
                                     LLM reads context + question → answer
   └────────────────────────┘        └──────────────────────────────────┘
```

The key idea: an **embedding** turns text into a point in space where *similar
meanings sit close together*. So "strawberries per hour" lands near the document
about the HarvestHop R9's picking rate, even if the wording differs. Retrieval is
just "find the nearest points," which here is a one-line numpy dot product
(the vectors are unit-length, so dot product = cosine similarity).

The LLM never learned the fact. You just handed it the right page to read.

---

## Now make it yours

1. **Your own knowledge base.** Replace the lines in `knowledge.jsonl` with your
   own `{"id": "...", "text": "..."}` documents — notes, product facts, a wiki
   page split into chunks. Re-run `02_build_index.py`, and ask away. (Try real
   facts the model *might* half-know, and watch RAG keep it accurate.)

2. **Knobs to turn:**
   - `TOP_K` in `03_ask_rag.py` / `04_chat_rag.py` — how many documents to
     retrieve. Too few misses context; too many adds noise. Try 1, 3, 5.
   - The `system_prompt` wording — this is prompt engineering (your specialty!).
     Tightening "use ONLY the context" reduces hallucination.
   - `EMBED_MODEL` in `rag_utils.py` — all-MiniLM-L6-v2 is the tiny starter.
     Bigger embedding models retrieve more accurately (and need a one-time
     download).

3. **Chunking.** Real documents are long. The standard move is to split them into
   ~100–300 word chunks before embedding, so retrieval returns a focused passage
   rather than a whole document. Your `knowledge.jsonl` is already "pre-chunked"
   into one fact per line — that's the idea in miniature.

---

## Troubleshooting

- **Timeout / connection error:** you skipped the one-time embedding-model
  download, or it didn't finish. Turn VPN on and run
  `hf download sentence-transformers/all-MiniLM-L6-v2` again.
- **Retrieved the wrong document:** small embedding models aren't perfect. Raise
  `TOP_K`, or write more distinct documents. This is a real, tunable part of RAG.
- **Model still hallucinates with context present:** tighten the system prompt,
  or lower `max_new_tokens`. 360M is small; bigger LLMs follow "use only the
  context" more faithfully.

---

## Fine-tuning vs RAG — when to use which

You now have both tools. The rule of thumb:

| | Fine-tuning | RAG |
|---|---|---|
| Changes | the model's **weights** | the model's **prompt** (context) |
| Best for | **behavior**: tone, format, style | **facts**: new, changing, or private knowledge |
| New facts? | unreliable | its whole purpose |
| Update a fact | retrain | edit a text file, re-index |
| Cost at answer time | none (baked in) | retrieval + bigger prompt |

They combine well: fine-tune for *how* the model should talk, use RAG for *what*
it should know. That pairing is how most real production LLM systems are built.
