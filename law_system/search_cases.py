"""
STEP 2 -- Find similar cases (retrieval only, no LLM).

Enter a situation in plain Chinese; it returns the most similar past verdicts
with their key metadata. This is the retrieval half -- fast, and it shows exactly
what the system found before any LLM gets involved.

Run:  python search_cases.py
"""

import json
import numpy as np
from embed_zh import embed_texts

TOP_K = 5

# Change this to any situation, or wrap in input() for interactive use.
SITUATION = "我朋友在2015年向我借了5万元，口头约定了利息，但没有写借条，现在他不还钱，我该怎么办？"

# Load the index built in step 1.
embeddings = np.load("law_embeddings.npy")
with open("law_metadata.json", "r", encoding="utf-8") as f:
    metadatas = json.load(f)

# Embed the query (is_query=True adds the bge instruction prefix) and score.
q = embed_texts([SITUATION], is_query=True)[0]
scores = embeddings @ q                      # cosine similarity (vectors are unit length)
top_idx = scores.argsort()[::-1][:TOP_K]

print(f"\n查询情况：{SITUATION}\n")
print(f"找到 {TOP_K} 个最相似的案例：\n")
for rank, i in enumerate(top_idx, 1):
    m = metadatas[i]
    print(f"{rank}. [{scores[i]:.3f}] {m.get('标题', '')}")
    print(f"   案号：{m.get('案号', '')} | 法院：{m.get('法院', '')}")
    print(f"   案由：{m.get('案由', '')} | 判决结果：{m.get('判决结果', '')}")
    print()
