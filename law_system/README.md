# Chinese legal case-search demo (RAG)

A working demo of the law system you described: a user types a situation in
Chinese, and the system finds similar past verdicts and returns reference
suggestions plus the older cases. It's RAG over your own CSV of court verdicts,
using a Chinese embedder for retrieval and a Chinese LLM for suggestions — all on
CPU, all offline once set up.

> ⚠️ This is a learning demo. Its output is for reference only, is **not legal
> advice**, and a real system needs a stronger model, careful evaluation, and a
> lawyer reviewing anything consequential.

---

## What's in this folder

| File | What it is |
|------|-----------|
| `requirements-law.txt` | Confirms your env + lists the two models to download |
| `embed_zh.py`          | Chinese embedder (bge-small-zh-v1.5) |
| `build_index.py`       | Step 1 — read your CSVs, embed every case, save the index |
| `search_cases.py`      | Step 2 — type a situation → get similar cases (no LLM) |
| `legal_assistant.py`   | Step 3 — similar cases + AI suggestions (the full demo) |

---

## Setup

Reuse your fine-tuning environment (no new packages needed — no FAISS, so no
segfault):

```bash
cd ~/Projects/fine-tuning && source .venv/bin/activate
# put this kit's files in a folder
mkdir data        # <-- put your verdict CSV files in here
```

Download the two models once. **In China, use ModelScope (no VPN):**

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; print(snapshot_download('BAAI/bge-small-zh-v1.5')); print(snapshot_download('Qwen/Qwen2.5-1.5B-Instruct'))"
```

ModelScope prints a local folder path for each model. Either:
- paste those paths into `EMBED_MODEL` (in `embed_zh.py`) and `LLM_ID`
  (in `legal_assistant.py`), **or**
- download the same ids via Hugging Face with a VPN, and the default ids work
  with your usual `HF_HUB_OFFLINE=1`.

---

## Run it

```bash
python build_index.py       # reads ./data/*.csv, embeds all cases, saves the index
python search_cases.py      # type a situation -> see similar cases (fast)
python legal_assistant.py   # similar cases + AI suggestions (slower, uses the LLM)
```

Edit the `SITUATION` variable at the top of the last two scripts to try different
queries (or wrap it in `input()` for interactive use).

---

## How it maps to your CSV

Your columns: `标题, 案号, 法院, 文书类型, 案件类型, 审理程序, 判决时间, 发布时间,
受理时间, 案例级别, 法院级别, 案由, 省, 市, 判决年份, 判决结果, 法条, 文书内容`.

- **Embedded for search:** `标题` + `案由` + `文书内容` (the facts). This is what
  gets matched against the user's situation.
- **Kept as metadata** (shown in results, fed to the LLM): `标题, 案号, 法院,
  案由, 判决结果, 法条, 判决年份, 省, 市`, plus a 300-char snippet of `文书内容`.

Two Chinese-CSV gotchas handled for you in `build_index.py`:
- **Encoding** — tries UTF-8 then GB18030 (many government exports are GB18030).
- **Field size** — `文书内容` is huge, so Python's CSV field limit is raised.

---

## How it works (recap of everything you've learned)

```
  user's situation (Chinese)
          │  embed (bge-zh, with query prefix)
          ▼
   query vector ──► cosine similarity vs all case vectors (numpy)
          │
          ▼  top-k similar verdicts
   retrieved cases ──► pasted into a Chinese prompt as context
          │
          ▼  Qwen2.5 reads cases + situation
   suggestions + case summaries (grounded in the retrieved verdicts)
```

It's the RAG pattern you built before, now in Chinese and over real data:
retrieval with a Chinese embedder, generation with a Chinese LLM, weights
untouched, knowledge living in your CSV files.

---

## Making it better (next steps)

1. **Chunking.** Right now each verdict is embedded as one (truncated) vector.
   Long verdicts lose their tail. The production move is to split each `文书内容`
   into ~300-character passages, embed each, and retrieve at the passage level —
   so a match points to the exact relevant paragraph, which also gives better
   citations.
2. **Metadata filtering.** Let the user restrict by `案由` (case type), `省`
   (province), or `判决年份` (year) before the similarity search — filter the
   metadata list, then score only those rows.
3. **A stronger model.** Qwen2.5-1.5B is small. On a GPU box you'd use
   Qwen2.5-7B or larger for more reliable suggestions; the code is the same
   one-line `LLM_ID` swap.
4. **Scale + speed.** At tens of thousands of verdicts, numpy is still fine; at
   hundreds of thousands, move retrieval to a vector database (FAISS on your
   WSL2 box, or a server like Qdrant) — the shape stays identical.
