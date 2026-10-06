#!/usr/bin/env python3
"""Train/test leakage check for the RAG-PIF datasets.

Per test set it reports:
  exact_norm      test texts identical to a training text after NFKC, zero-width
                  removal, whitespace collapse and lowercasing
  exact_deobf     same, after also mapping leetspeak (0->o, 1->i, 3->e, 4->a, 5->s, 7->t, @->a, $->s)
                  on both sides. Catches evasion variants of training injections
  near_dup@T      test texts whose nearest training text has MiniLM cosine >= T
  index@T         test injections (label 1) whose nearest FAISS-index vector has cosine >= T
                  (Layer 2 blocks above 0.50, so index@0.50 is the share Layer 2 can fire on)
  benign_topic@T  test benign texts (label 0) whose nearest training benign text has cosine >= T
                  (proxy for same-article/topic leakage; the datasets have no article ids)

Usage
  python evaluation/scripts/leakage_check.py
  python evaluation/scripts/leakage_check.py --train train_augmented.csv --train train.csv
"""
import argparse
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SETS = {
    "standard": "test.csv",
    "evasion": "test_evasion.csv",
    "adversarial": "test_adversarial.csv",
    "ood": "test_ood.csv",
}
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
NEAR_T = [0.80, 0.90, 0.95]
INDEX_T = [0.50, 0.90, 0.95]
TOPIC_T = [0.60, 0.70]


def norm(t):
    t = unicodedata.normalize("NFKC", str(t))
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")  # strips zero-width chars
    return re.sub(r"\s+", " ", t).strip().lower()


def deobf(t):
    return norm(t).translate(LEET)


def find_file(root, name):
    root = Path(root)
    if (root / name).exists():
        return root / name
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def frac(mask):
    return float(np.mean(mask)) if len(mask) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--faiss", default=str(REPO / "datasets" / "injection_index.faiss"))
    ap.add_argument("--train", action="append", default=None,
                    help="training CSV name(s) under --data-dir (repeatable). Default: train_augmented.csv")
    ap.add_argument("--out-dir", default=str(REPO / "evaluation" / "runs"))
    args = ap.parse_args()

    import faiss
    from sentence_transformers import SentenceTransformer

    train_names = args.train or ["train_augmented.csv"]
    train = pd.concat([pd.read_csv(find_file(args.data_dir, n)).assign(_src=n) for n in train_names],
                      ignore_index=True)
    train["text"] = train["text"].astype(str)
    print(f"training texts: {len(train)} from {train_names}")

    tr_norm = set(train["text"].map(norm))
    tr_deobf = set(train["text"].map(deobf))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda xs: model.encode(list(xs), normalize_embeddings=True, batch_size=64, show_progress_bar=False).astype("float32")
    tr_emb = enc(train["text"])
    tr_lab = train["label"].astype(int).to_numpy()
    index = faiss.read_index(str(args.faiss))

    run_dir = Path(args.out_dir) / f"leakage_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, fname in SETS.items():
        df = pd.read_csv(find_file(args.data_dir, fname))
        texts = df["text"].astype(str).tolist()
        y = df["label"].astype(int).to_numpy()
        emb = enc(texts)

        row = {"test_set": name, "n": len(df)}
        row["exact_norm"] = int(sum(norm(t) in tr_norm for t in texts))
        row["exact_deobf"] = int(sum(deobf(t) in tr_deobf for t in texts))

        sims = emb @ tr_emb.T
        nn_all = sims.max(axis=1)
        for T in NEAR_T:
            row[f"near_dup@{T:.2f}"] = int((nn_all >= T).sum())

        inj = y == 1
        if inj.any():
            D, _ = index.search(emb[inj], 1)
            for T in INDEX_T:
                row[f"index@{T:.2f}"] = frac(D[:, 0] >= T)

        ben = y == 0
        if ben.any() and (tr_lab == 0).any():
            nn_ben = sims[np.ix_(ben, tr_lab == 0)].max(axis=1)
            for T in TOPIC_T:
                row[f"benign_topic@{T:.2f}"] = frac(nn_ben >= T)

        rows.append(row)

        nn_idx = sims.argmax(axis=1)
        pairs = pd.DataFrame({
            "similarity": nn_all,
            "test_label": y,
            "test_text": texts,
            "nearest_train_text": train["text"].to_numpy()[nn_idx],
            "nearest_train_label": tr_lab[nn_idx],
        }).sort_values("similarity", ascending=False).head(50)
        pairs.to_csv(run_dir / f"closest_pairs_{name}.csv", index=False)

    report = pd.DataFrame(rows)
    report.to_csv(run_dir / "leakage_report.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n" + report.round(3).to_string(index=False))

    print("\nread this as:")
    for r in rows:
        n = r["n"]
        flags = []
        if r["exact_deobf"] / n > 0.05:
            flags.append(f"{r['exact_deobf']}/{n} are exact matches of training text after de-obfuscation")
        if r["near_dup@0.95"] / n > 0.10:
            flags.append(f"{r['near_dup@0.95']}/{n} have a >=0.95 near-duplicate in training")
        if r.get("index@0.90", 0) and r["index@0.90"] > 0.30:
            flags.append(f"{r['index@0.90']:.0%} of injections sit within 0.90 of the FAISS index")
        print(f"  {r['test_set']:<12} " + ("LIKELY LEAKAGE: " + "; ".join(flags) if flags else "no strong leakage flag"))
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
