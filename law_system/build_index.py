"""
STEP 1 -- Read your verdict CSVs and build the search index.

Put your CSV files in ./data/ . This reads them all, builds one searchable text
per case, embeds them, and saves the index + metadata to disk. Run once (re-run
when you add more CSVs).

Handles two Chinese-government-CSV gotchas:
  * encoding: tries UTF-8 then GB18030 (many gov files are GB18030/GBK).
  * huge fields: the 文书内容 column overflows Python's default CSV limit.

Run:  python build_index.py
"""

import os
import sys
import csv
import glob
import json
import numpy as np
from embed_zh import embed_texts, EMBED_DIM

csv.field_size_limit(sys.maxsize)   # legal document text is very long

DATA_DIR = "data"
# Fields we keep for display (everything except the giant 文书内容).
META_FIELDS = ["标题", "案号", "法院", "案件类型", "审理程序", "判决时间",
               "案由", "省", "市", "判决年份", "判决结果", "法条"]


def read_all_rows():
    rows = []
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "202411.csv")))
    if not paths:
        print(f"No CSV files found in {DATA_DIR}/ -- put your verdict CSVs there.")
        sys.exit(1)
    for path in paths:
        for enc in ("utf-8-sig", "gb18030"):          # try UTF-8, then GB18030
            try:
                with open(path, newline="", encoding=enc) as f:
                    file_rows = list(csv.DictReader(f))
                print(f"  {path}: {len(file_rows)} rows (encoding: {enc})")
                rows.extend(file_rows)
                break
            except UnicodeDecodeError:
                continue
    return rows


def build_search_text(row):
    """What we embed: title + cause of action + the document body (facts)."""
    return f"{row.get('标题', '')}。案由：{row.get('案由', '')}。{row.get('文书内容', '')}"


def main():
    rows = read_all_rows()
    print(f"\nTotal cases: {len(rows)}")

    search_texts = [build_search_text(r) for r in rows]
    metadatas = [{k: r.get(k, "") for k in META_FIELDS} for r in rows]
    # keep a short snippet of the body for the LLM to read later
    for meta, r in zip(metadatas, rows):
        meta["摘要"] = (r.get("文书内容", "") or "")[:300]

    print("Embedding cases...")
    embeddings = embed_texts(search_texts, is_query=False)

    np.save("law_embeddings.npy", embeddings)
    with open("law_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False)

    print(f"Saved index for {len(rows)} cases (dim={EMBED_DIM}).")
    print("Next: python search_cases.py")


if __name__ == "__main__":
    main()
