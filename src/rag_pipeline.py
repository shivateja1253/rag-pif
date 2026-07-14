import re
import unicodedata
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

INJECTION_PATTERNS = [
    r"ignore\s+(\w+\s+)*(instructions?|prompt|context|rules?)",
    r"forget\s+(everything|all|previous|your\s+instructions)",
    r"disregard\s+(the\s+|all\s+|previous\s+)?(above|instructions?|context)",
    r"you\s+are\s+now\s+(?!an?\s+assistant)",
    r"act\s+as\s+(?!an?\s+assistant)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"(DAN|jailbreak|unrestricted)\s+mode",
    r"no\s+(restrictions?|filters?|limits?)",
]

THRESHOLD = 0.45

def layer1_filter(text):
    normalized = unicodedata.normalize("NFKC", text).lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized):
            return {"blocked": True, "layer": 1}
    return {"blocked": False, "layer": 1}

def layer2_filter(text, model, injection_index):
    vec = model.encode([text], normalize_embeddings=True).astype("float32")
    D, _ = injection_index.search(vec, k=1)
    score = float(D[0][0])
    return {"blocked": score > THRESHOLD, "layer": 2, "score": round(score, 4)}

def rag_with_firewall(query, chunks, retrieval_index, injection_index, model, top_k=3):
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = retrieval_index.search(q_vec, k=top_k*2)
    retrieved = [chunks[i] for i in I[0] if i < len(chunks)]
    safe_chunks, blocked_chunks = [], []
    for chunk in retrieved:
        r1 = layer1_filter(chunk)
        if r1["blocked"]:
            blocked_chunks.append((chunk[:80], "Layer 1"))
            continue
        r2 = layer2_filter(chunk, model, injection_index)
        if r2["blocked"]:
            blocked_chunks.append((chunk[:80], f"Layer 2 score={r2['score']}"))
            continue
        safe_chunks.append(chunk)
    return safe_chunks[:top_k], blocked_chunks
