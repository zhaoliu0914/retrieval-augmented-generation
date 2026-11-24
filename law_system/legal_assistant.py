"""
STEP 3 -- The full demo: similar cases + AI suggestions.

Retrieves similar past verdicts, then asks Qwen2.5 to (a) give reference
suggestions grounded in those cases and (b) briefly summarize the similar cases.
All in Chinese.

IMPORTANT: this is a DEMO. The output is for reference only, is not legal advice,
and the system prompt tells the model to say so. A real system needs a stronger
model, careful review, and a lawyer in the loop.

Run:  python legal_assistant.py
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from embed_zh import embed_texts

LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct"    # or a local path from ModelScope
TOP_K = 3

SITUATION = "我朋友在2015年向我借了5万元，口头约定了利息，但没有写借条，现在他不还钱，我该怎么办？"

# --- 1. RETRIEVE similar cases ---
embeddings = np.load("law_embeddings.npy")
with open("law_metadata.json", "r", encoding="utf-8") as f:
    metadatas = json.load(f)

q = embed_texts([SITUATION], is_query=True)[0]
scores = embeddings @ q
top_idx = scores.argsort()[::-1][:TOP_K]
hits = [metadatas[i] for i in top_idx]

# --- 2. AUGMENT: build Chinese context from the retrieved cases ---
case_blocks = []
for n, m in enumerate(hits, 1):
    case_blocks.append(
        f"案例{n}：{m.get('标题','')}\n"
        f"案由：{m.get('案由','')}\n"
        f"判决结果：{m.get('判决结果','')}\n"
        f"相关法条：{m.get('法条','')}\n"
        f"案情摘要：{m.get('摘要','')}"
    )
context = "\n\n".join(case_blocks)

system_prompt = (
    "你是一个法律信息检索助手。请根据下面检索到的相似判例，为用户的情况提供参考建议，"
    "输出相似案例完整的事实，理由，辩论焦点，和判决结果。"
    "只能依据提供的案例内容作答，不要编造法律条文或事实。"
    "请在回答开头声明：本回答仅供参考，不构成正式法律意见，具体问题请咨询专业律师。\n\n"
    f"检索到的相似判例：\n{context}"
)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"我的情况：{SITUATION}\n请给出参考建议，并介绍相似案例。"},
]

# --- 3. GENERATE ---
print("Loading Qwen2.5-1.5B...")
tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
model = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=torch.float32)
model.eval()

inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
)
print("Generating (CPU -- this can take a few minutes)...")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=512, do_sample=True)

reply = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print("\n" + "=" * 60)
print(f"用户情况：{SITUATION}")
print("=" * 60)
print("\n【检索到的相似案例】")
for n, m in enumerate(hits, 1):
    print(f"  {n}. {m.get('标题','')}（{m.get('案号','')}）")
print("\n【AI 参考建议】")
print(reply)
