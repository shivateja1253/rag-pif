#!/usr/bin/env python3
"""Build leakage-resistant train/val/test splits for RAG-PIF (dataset v2.1).

Why this exists
  The original splits leak: every evasion test sample is an obfuscated copy of a
  HackaPrompt injection that also sits in train and in the Layer 2 FAISS index,
  and benign Wikipedia chunks from the same articles land on both sides.

What it does
  1. Pool = rag_inject_bench_v2.csv minus the derived evasion rows
     (categories unicode / obfuscated / zero_width). Evasion is regenerated below.
  2. Drops exact duplicates (after NFKC, zero-width removal, whitespace and case folding).
  3. Groups near-duplicates: injections by char n-gram TF-IDF cosine >= --inj-thr,
     benign chunks by word TF-IDF cosine >= --benign-thr (proxy for "same article",
     since the data has no article ids). Connected components = groups.
  4. Assigns whole groups to train/val/test, stratified by label|category.
  5. Generates evasion variants (fullwidth, leetspeak, zero-width) ONLY from the
     English 'direct' injections of each split, so a test variant's base is never in train.
  6. Validates: no exact key and no group in two splits; no cross-split pair above
     the grouping thresholds; reports the same for the manual adversarial/OOD sets.

Usage
  python scripts/build_splits.py --data-dir data --out-dir data/v2_1 --seed 42
"""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

DERIVED = {"unicode", "obfuscated", "zero_width"}
SPLITS = ["train", "val", "test"]
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
LEET_FWD = str.maketrans({"i": "1", "o": "0", "e": "3", "a": "4", "s": "5", "t": "7",
                          "I": "1", "O": "0", "E": "3", "A": "4", "S": "5", "T": "7"})


# ------------------------------------------------------------ normalization
def norm(t):
    t = unicodedata.normalize("NFKC", str(t))
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")  # zero-width etc.
    return re.sub(r"\s+", " ", t).strip().lower()


def key(t, leet):
    s = norm(t)
    if leet:
        s = s.translate(LEET)
    return re.sub(r"[^\w]+", "", s)


# --------------------------------------------------------- evasion transforms
def to_fullwidth(t):
    return "".join(chr(ord(c) + 0xFEE0) if 0x21 <= ord(c) <= 0x7E else c for c in t)


def to_leet(t):
    return t.translate(LEET_FWD)


def to_zero_width(t):
    return t.replace(" ", "\u200b")


EVASION = {"unicode": to_fullwidth, "obfuscated": to_leet, "zero_width": to_zero_width}


def make_evasion(bases):
    rows = []
    for _, r in bases.iterrows():
        for cat, fn in EVASION.items():
            v = fn(r["text"])
            if v != r["text"]:
                rows.append({"text": v, "label": 1, "category": cat, "source": "synthetic_evasion",
                             "base_group": r["group"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ grouping
def cluster(texts, analyzer, thr):
    """Connected components of the cosine>=thr graph. Returns component label per text."""
    if len(texts) == 0:
        return np.array([], dtype=int)
    if analyzer == "char":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    else:
        vec = TfidfVectorizer(sublinear_tf=True, stop_words="english")
    X = vec.fit_transform([norm(t) for t in texts])
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0.0)
    _, lab = connected_components(csr_matrix(S >= thr), directed=False)
    return lab


def assign_groups(groups, targets, seed):
    """groups: DataFrame[group, stratum, size]. Greedy fill toward per-stratum targets."""
    rng = np.random.default_rng(seed)
    out = {}
    for stratum, g in groups.groupby("stratum"):
        total = g["size"].sum()
        goal = {s: targets[s] * total for s in SPLITS}
        have = {s: 0 for s in SPLITS}
        order = g.assign(r=rng.random(len(g))).sort_values(["size", "r"], ascending=[False, True])
        for _, row in order.iterrows():
            best = max(SPLITS, key=lambda s: (goal[s] - have[s]) / max(goal[s], 1e-9))
            out[row["group"]] = best
            have[best] += row["size"]
    return out


# ---------------------------------------------------------------- validation
def max_cross_sim(a_texts, b_texts, analyzer):
    """Max cosine of each a-text to any b-text, TF-IDF fit on both."""
    if not len(a_texts) or not len(b_texts):
        return np.array([])
    vec = (TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
           if analyzer == "char" else TfidfVectorizer(sublinear_tf=True, stop_words="english"))
    X = vec.fit_transform([norm(t) for t in list(a_texts) + list(b_texts)])
    A, B = X[: len(a_texts)], X[len(a_texts):]
    return (A @ B.T).toarray().max(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="data/v2_1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inj-thr", type=float, default=0.85, help="char n-gram cosine for injection groups")
    ap.add_argument("--benign-thr", type=float, default=0.25, help="word TF-IDF cosine for benign groups")
    ap.add_argument("--targets", default="0.70,0.15,0.15")
    ap.add_argument("--hard-inj", type=float, default=0.90, help="validation: injection pairs this similar must not cross splits")
    ap.add_argument("--hard-benign", type=float, default=0.35, help="validation: benign pairs this similar must not cross splits")
    a = ap.parse_args()

    targets = dict(zip(SPLITS, [float(x) for x in a.targets.split(",")]))
    data, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pool_raw = pd.read_csv(data / "rag_inject_bench_v2.csv")
    pool = pool_raw[~pool_raw["category"].isin(DERIVED)].copy().reset_index(drop=True)
    report = {"seed": a.seed, "thresholds": {"inj": a.inj_thr, "benign": a.benign_thr},
              "pool_rows_raw": len(pool_raw), "derived_evasion_rows_removed": int(len(pool_raw) - len(pool))}

    # --- exact-duplicate removal (per label, so label conflicts stay visible)
    pool["key"] = [key(t, leet=(l == 1)) for t, l in zip(pool["text"], pool["label"])]
    conflicts = set(pool[pool.label == 1]["key"]) & set(pool[pool.label == 0]["key"])
    dups = pool[pool.duplicated(["label", "key"], keep="first")]
    dups.drop(columns=["key"]).to_csv(out / "dropped_duplicates.csv", index=False)
    pool = pool.drop_duplicates(["label", "key"], keep="first").reset_index(drop=True)
    report["exact_duplicates_dropped"] = int(len(dups))
    report["label_conflicts"] = len(conflicts)

    # --- near-duplicate grouping, separately per label
    pool["group"] = ""
    for label, analyzer, thr, prefix in [(1, "char", a.inj_thr, "I"), (0, "word", a.benign_thr, "B")]:
        idx = pool.index[pool.label == label]
        lab = cluster(pool.loc[idx, "text"].tolist(), analyzer, thr)
        pool.loc[idx, "group"] = [f"{prefix}{x:04d}" for x in lab]

    # --- stratified group assignment
    def majority(s):
        return s.value_counts().index[0]

    groups = (pool.groupby("group")
              .agg(label=("label", "first"), category=("category", majority), size=("text", "size"))
              .reset_index())
    groups["stratum"] = groups["label"].astype(str) + "|" + groups["category"]
    pool["split"] = pool["group"].map(assign_groups(groups, targets, a.seed))
    report["groups"] = {
        "injection_groups": int((groups.label == 1).sum()), "benign_groups": int((groups.label == 0).sum()),
        "largest_injection_group": int(groups[groups.label == 1]["size"].max()),
        "largest_benign_group": int(groups[groups.label == 0]["size"].max()),
    }

    # --- write splits
    cols = ["text", "label", "category", "source"]
    for s in SPLITS:
        pool[pool.split == s][cols + ["group"]].to_csv(out / f"{s}.csv", index=False)
    pool[["split", "group", "label", "category", "source"]].assign(
        text_sha1=[hashlib.sha1(t.encode("utf-8")).hexdigest() for t in pool["text"]]
    ).to_csv(out / "split_manifest.csv", index=False)

    # --- evasion variants from each split's own English 'direct' injections
    ev_counts = {}
    for s in SPLITS:
        bases = pool[(pool.split == s) & (pool.label == 1) & (pool.category == "direct")]
        ev = make_evasion(bases)
        name = {"train": "train_evasion.csv", "val": "val_evasion.csv", "test": "test_evasion.csv"}[s]
        ev.to_csv(out / name, index=False)
        ev_counts[s] = {"bases": int(len(bases)), "variants": int(len(ev))}
    report["evasion"] = ev_counts

    # --- counts
    report["rows_per_split"] = {s: {
        "total": int((pool.split == s).sum()),
        "injection": int(((pool.split == s) & (pool.label == 1)).sum()),
        "benign": int(((pool.split == s) & (pool.label == 0)).sum())} for s in SPLITS}
    report["category_counts"] = {s: pool[pool.split == s]["category"].value_counts().to_dict() for s in SPLITS}

    # --- validation (independent re-check across splits)
    checks = {}
    for s in ["val", "test"]:
        k_tr = set(zip(pool[pool.split == "train"]["label"], pool[pool.split == "train"]["key"]))
        k_s = set(zip(pool[pool.split == s]["label"], pool[pool.split == s]["key"]))
        checks[f"exact_key_overlap_train_vs_{s}"] = len(k_tr & k_s)
        for label, analyzer, thr, nm in [(1, "char", a.inj_thr, "injection"), (0, "word", a.benign_thr, "benign")]:
            A = pool[(pool.split == s) & (pool.label == label)]["text"].tolist()
            B = pool[(pool.split == "train") & (pool.label == label)]["text"].tolist()
            m = max_cross_sim(A, B, analyzer)
            hard = a.hard_inj if label == 1 else a.hard_benign
            checks[f"{nm}_{s}_pairs_above_hard_threshold_vs_train"] = int((m >= hard).sum())
            checks[f"{nm}_{s}_borderline_pairs_above_group_threshold_vs_train"] = int((m >= thr).sum())
            checks[f"{nm}_{s}_max_sim_to_train_p95"] = float(np.quantile(m, 0.95)) if len(m) else None
    tr_keys = set(pool[pool.split == "train"]["key"])
    for s in ["val", "test"]:
        ev = pd.read_csv(out / f"{s}_evasion.csv")
        checks[f"{s}_evasion_deobf_key_in_train"] = int(sum(key(t, True) in tr_keys for t in ev["text"]))

    # --- manual challenge sets: independence from the NEW train split
    tr_inj = pool[(pool.split == "train") & (pool.label == 1)]["text"].tolist()
    tr_ben = pool[(pool.split == "train") & (pool.label == 0)]["text"].tolist()
    chal = {}
    for fname, label in [("test_adversarial.csv", 0), ("test_ood.csv", 0)]:
        p = data / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        texts = df["text"].astype(str).tolist()
        m_tr = max_cross_sim(texts, tr_ben, "word")
        m_self = []
        if len(texts) > 1:
            vec = TfidfVectorizer(sublinear_tf=True, stop_words="english")
            X = vec.fit_transform([norm(t) for t in texts])
            S = (X @ X.T).toarray()
            np.fill_diagonal(S, 0.0)
            m_self = S.max(axis=1)
        chal[fname] = {
            "n": len(df),
            "exact_in_new_train": int(sum(key(t, False) in set(pool[(pool.split == 'train')]['key']) for t in texts)),
            "share_word_cosine_ge_0.5_to_train_benign": float(np.mean(m_tr >= 0.5)),
            "share_with_in_set_neighbor_ge_0.5": float(np.mean(np.array(m_self) >= 0.5)) if len(m_self) else None,
        }
        df.to_csv(out / fname, index=False)
    report["manual_sets"] = chal

    def leak(tr_df, te_df, label, analyzer, ts):
        A = te_df[te_df.label == label]["text"].astype(str).tolist()
        B = tr_df[tr_df.label == label]["text"].astype(str).tolist()
        m = max_cross_sim(A, B, analyzer)
        return {"n": len(A), **{f">={t}": float(np.mean(m >= t)) for t in ts}} if len(m) else {}

    old = {}
    if (data / "train.csv").exists() and (data / "test.csv").exists():
        otr, ote = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
        old["standard_injections_vs_train (char cosine)"] = leak(otr, ote, 1, "char", [0.85, 0.95])
        old["standard_benign_vs_train (word cosine)"] = leak(otr, ote, 0, "word", [0.25, 0.35])
        if (data / "test_evasion.csv").exists():
            oev = pd.read_csv(data / "test_evasion.csv")
            old["evasion_vs_train (char cosine, after NFKC/zero-width/leet normalization)"] = {
                "n": len(oev), **{f">={t}": float(np.mean(max_cross_sim(
                    [norm(x).translate(LEET) for x in oev["text"]],
                    [norm(x).translate(LEET) for x in otr[otr.label == 1]["text"]], "char") >= t)) for t in [0.85, 0.95]}}
    nw = {}
    ntr, nte = pool[pool.split == "train"], pool[pool.split == "test"]
    nw["standard_injections_vs_train (char cosine)"] = leak(ntr, nte, 1, "char", [0.85, 0.95])
    nw["standard_benign_vs_train (word cosine)"] = leak(ntr, nte, 0, "word", [0.25, 0.35])
    nev = pd.read_csv(out / "test_evasion.csv")
    nw["evasion_vs_train (char cosine, after NFKC/zero-width/leet normalization)"] = {
        "n": len(nev), **{f">={t}": float(np.mean(max_cross_sim(
            [norm(x).translate(LEET) for x in nev["text"]],
            [norm(x).translate(LEET) for x in ntr[ntr.label == 1]["text"]], "char") >= t)) for t in [0.85, 0.95]}}
    report["leakage_old_splits"] = old
    report["leakage_new_splits"] = nw
    report["validation"] = checks
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    bad = {k: v for k, v in checks.items() if (k.startswith("exact") or "above_hard" in k or "in_train" in k) and v}
    print("\nVALIDATION:", "PASS" if not bad else f"FAIL {bad}")
    print("wrote", out)


if __name__ == "__main__":
    main()
