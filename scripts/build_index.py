#!/usr/bin/env python3
"""Build the Layer 2 FAISS index from TRAIN injections only.

The v1 index was built from all 550 v1 injection rows, which include every
evasion test sample and every v1-category standard test injection verbatim.
This script indexes only the training split, and saves the indexed texts next to
the index so the index can be audited (v1 could not be mapped back to texts).

    python scripts/build_index.py --train data/v2_1/train.csv --out datasets/injection_index_v2_1.faiss
"""
import argparse
import hashlib
import json
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/v2_1/train.csv")
    ap.add_argument("--out", default="datasets/injection_index_v2_1.faiss")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    a = ap.parse_args()

    df = pd.read_csv(a.train)
    inj = df[df["label"] == 1]["text"].astype(str).tolist()
    model = SentenceTransformer(a.model)
    # raw text, normalized embeddings, inner product: same as the Layer 2 scoring path
    emb = model.encode(inj, normalize_embeddings=True, batch_size=64).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))
    pd.DataFrame({"text": inj}).to_csv(out.with_suffix(".texts.csv"), index=False)
    meta = {"n_vectors": index.ntotal, "dim": index.d, "embedding_model": a.model,
            "train_file": a.train, "train_sha256": hashlib.sha256(Path(a.train).read_bytes()).hexdigest()}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
