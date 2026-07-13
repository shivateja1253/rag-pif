import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model and index
model = SentenceTransformer("all-MiniLM-L6-v2")
THRESHOLD = 0.45

def load_faiss_index(path: str):
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found at {path}")
    return faiss.read_index(path)

def layer2_classifier(text: str, index) -> dict:
    if index.ntotal == 0:
        return {"blocked": False, "layer": 2, "confidence": 0.0}
    vec = model.encode([text], normalize_embeddings=True).astype("float32")
    D, I = index.search(vec, k=1)
    score = float(D[0][0])
    return {"blocked": score > THRESHOLD, "layer": 2, "confidence": round(score, 4)}
