"""
rag_firewall.py -- Full 3-layer prompt injection firewall pipeline.

This file was referenced in project documentation throughout Month 1-3
but never actually existed in the repository until this Month 3 push.
Both the Streamlit demo (src/pages/1_Chatbot.py) and the evaluation
benchmark implement this same L1 -> L2 -> L3 cascade logic; this module
is the canonical, importable version of it.

Usage:
    from rag_firewall import RagFirewall

    firewall = RagFirewall(
        faiss_index_path="data/injection_index.faiss",
        roberta_model_dir="models/roberta_v2_hf",
        roberta_tokenizer_dir="models/roberta_tokenizer_v2",
    )
    result = firewall.check("some text")
    # result = {"blocked": bool, "layer": 1|2|3|None, "confidence": float, "detail": str|None}
"""

import time
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

from layer1_filter import layer1_filter
from layer2_embedding import THRESHOLD as L2_THRESHOLD

L3_THRESHOLD = 0.50


class RagFirewall:
    def __init__(self, faiss_index_path, roberta_model_dir, roberta_tokenizer_dir, device=None):
        self.l2_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.l2_index = faiss.read_index(faiss_index_path)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.l3_tokenizer = RobertaTokenizerFast.from_pretrained(roberta_tokenizer_dir)
        self.l3_model = RobertaForSequenceClassification.from_pretrained(roberta_model_dir)
        self.l3_model.to(self.device)
        self.l3_model.eval()

    def _layer2(self, text: str) -> dict:
        if self.l2_index.ntotal == 0:
            return {"blocked": False, "confidence": 0.0}
        vec = self.l2_model.encode([text], normalize_embeddings=True).astype("float32")
        D, _ = self.l2_index.search(vec, k=1)
        score = float(D[0][0])
        return {"blocked": score > L2_THRESHOLD, "confidence": round(score, 4)}

    @torch.no_grad()
    def _layer3(self, text: str) -> dict:
        inputs = self.l3_tokenizer(text, return_tensors="pt", truncation=True,
                                    max_length=256, padding=True).to(self.device)
        logits = self.l3_model(**inputs).logits
        prob = float(torch.softmax(logits, dim=-1)[0][1])
        return {"blocked": prob > L3_THRESHOLD, "confidence": round(prob, 4)}

    def check(self, text: str) -> dict:
        """Runs the full L1 -> L2 -> L3 cascade. Stops at the first layer that blocks."""
        t0 = time.perf_counter()

        r1 = layer1_filter(text)
        if r1["blocked"]:
            return {"blocked": True, "layer": 1, "confidence": r1["confidence"],
                    "detail": f'Pattern: "{r1["pattern"]}"',
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}

        r2 = self._layer2(text)
        if r2["blocked"]:
            return {"blocked": True, "layer": 2, "confidence": r2["confidence"],
                    "detail": f'Similarity score: {r2["confidence"]}',
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}

        r3 = self._layer3(text)
        return {"blocked": r3["blocked"], "layer": 3 if r3["blocked"] else None,
                "confidence": r3["confidence"],
                "detail": f'RoBERTa confidence: {r3["confidence"]}' if r3["blocked"] else None,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}
