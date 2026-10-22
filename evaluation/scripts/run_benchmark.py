#!/usr/bin/env python3
"""Local, reproducible RAG-PIF benchmark runner (replaces the Colab-only script).

What it does
  * scores every sample with all three layers once, stores the raw scores
  * derives the cascade decision and every layer ablation (L1, L2, L3, pairs, all)
    from those scores, so ablations cost nothing extra
  * reports 95% Wilson confidence intervals
  * records file hashes, library versions, git commit, borderline-score counts
  * --repeat N re-scores N times and reports the max score drift (determinism)
  * --expect compares against a previous results CSV (e.g. FINAL_benchmark_results.csv)

Layer 1 is imported from streamlit/layer1_filter.py, so there is one source of
truth. Latency is not measured here (all layers run on every sample).

Examples
  python evaluation/scripts/run_benchmark.py --tag prod --repeat 2 \
      --expect evaluation/FINAL_benchmark_results.csv
  python evaluation/scripts/run_benchmark.py --tag v3 --l3-subfolder experiments/v3
  python evaluation/scripts/run_benchmark.py --tag local --l3 models/roberta_v2_hf \
      --tokenizer models/roberta_tokenizer_v2
"""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "streamlit"))
sys.path.insert(0, str(REPO / "scripts"))
from layer1_filter import layer1_filter  # noqa: E402  single source of truth for Layer 1
from textnorm import prenorm  # noqa: E402

SETS = {
    "standard": "test.csv",
    "evasion": "test_evasion.csv",
    "adversarial": "test_adversarial.csv",
    "ood": "test_ood.csv",
}
L2_THRESHOLD = 0.50
L3_THRESHOLD = 0.50
COMBOS = ["L1", "L2", "L3", "L1+L2", "L1+L3", "L2+L3", "L1+L2+L3"]


# ----------------------------------------------------------------- utilities
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_file(root, name):
    root = Path(root)
    direct = root / name
    if direct.exists():
        return direct
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    pos, neg = tp + fn, tn + fp
    out = {"n": int(len(y_true)), "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    if neg:
        lo, hi = wilson(fp, neg)
        out.update(fpr=fp / neg, fpr_ci_lo=lo, fpr_ci_hi=hi)
    if pos:
        lo, hi = wilson(tp, pos)
        out.update(detection_rate=tp / pos, det_ci_lo=lo, det_ci_hi=hi)
    if pos and neg:
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / pos
        out.update(precision=prec, recall=rec, f1=(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0)
    return out


def derive(scores):
    """scores: dict with b1 (bool arr), s2 (float arr), p3 (float arr).
    Returns (per-combo boolean decisions, first-blocking-layer array)."""
    b1 = scores["b1"].astype(bool)
    b2 = scores["s2"] > L2_THRESHOLD
    b3 = scores["p3"] > L3_THRESHOLD
    combos = {
        "L1": b1, "L2": b2, "L3": b3,
        "L1+L2": b1 | b2, "L1+L3": b1 | b3, "L2+L3": b2 | b3,
        "L1+L2+L3": b1 | b2 | b3,
    }
    first = np.where(b1, 1, np.where(b2, 2, np.where(b3, 3, 0)))
    return combos, first


def cascade_row(name, y, first):
    m = metrics(y, first > 0)
    m.update(
        test_set=name,
        blocked_by_l1=int((first == 1).sum()),
        blocked_by_l2=int((first == 2).sum()),
        blocked_by_l3=int((first == 3).sum()),
    )
    return m


# ------------------------------------------------------------------- models
def load_models(args):
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss
    from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    l2_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    index = faiss.read_index(str(args.faiss))
    tok = l3 = None
    if not args.no_l3:
        tok = RobertaTokenizerFast.from_pretrained(args.tokenizer or args.l3)
        l3 = RobertaForSequenceClassification.from_pretrained(args.l3, subfolder=args.l3_subfolder or "")
        l3.to(device).eval()
    return {"torch": torch, "device": device, "l2": l2_model, "index": index, "tok": tok, "l3": l3}


def score_texts(texts, m):
    torch = m["torch"]
    b1, s2, p3 = [], [], []
    for t in texts:
        b1.append(bool(layer1_filter(t)["blocked"]))
        v = m["l2"].encode([t], normalize_embeddings=True).astype("float32")
        D, _ = m["index"].search(v, 1)
        s2.append(float(D[0][0]))
        if m["l3"] is None:
            p3.append(0.0)  # --no-l3: Layer 3 disabled, L3 columns in the output are meaningless
            continue
        enc = m["tok"](t, return_tensors="pt", truncation=True, max_length=256, padding=True).to(m["device"])
        with torch.no_grad():
            logits = m["l3"](**enc).logits
        p3.append(float(torch.softmax(logits, dim=-1)[0][1]))
    return {"b1": np.array(b1), "s2": np.array(s2), "p3": np.array(p3)}


# --------------------------------------------------------------- provenance
def git_info():
    def run(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def hub_sha(repo_id):
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(repo_id).sha
    except Exception:
        return None


def lib_versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ["numpy", "pandas", "torch", "transformers", "sentence_transformers", "faiss"]:
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


# --------------------------------------------------------------------- main
def compare_expected(path, rows):
    exp = pd.read_csv(path).set_index("test_set")
    cols = ["tp", "fp", "tn", "fn", "blocked_by_l1", "blocked_by_l2", "blocked_by_l3"]
    report = {}
    for r in rows:
        name = r["test_set"]
        if name not in exp.index:
            report[name] = "no expected row"
            continue
        diffs = {}
        for c in cols:
            if c in exp.columns and pd.notna(exp.loc[name, c]):
                e, g = int(exp.loc[name, c]), int(r[c])
                if e != g:
                    diffs[c] = {"expected": e, "got": g}
        report[name] = "MATCH" if not diffs else diffs
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--faiss", default=str(REPO / "datasets" / "injection_index.faiss"))
    ap.add_argument("--l3", default="shivateja1253/rag-pif-roberta", help="HF repo id or local dir")
    ap.add_argument("--l3-subfolder", default="", help="e.g. experiments/v3")
    ap.add_argument("--tokenizer", default=None, help="defaults to --l3 (root tokenizer)")
    ap.add_argument("--no-l3", action="store_true", help="skip Layer 3 (evaluate L1+L2 only)")
    ap.add_argument("--prenorm", choices=["none", "full"], default="none",
                    help="normalization front-end applied to the text before scoring (scripts/textnorm.py)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda (cpu is most deterministic)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out-dir", default=str(REPO / "evaluation" / "runs"))
    ap.add_argument("--expect", default=None, help="results CSV to diff against")
    args = ap.parse_args()

    t0 = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_dir) / f"{args.tag}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    hashes = {"faiss_index": sha256_file(args.faiss)}
    for name, fname in SETS.items():
        p = find_file(args.data_dir, fname)
        df = pd.read_csv(p)
        data[name] = df
        hashes[fname] = sha256_file(p)

    models = load_models(args)
    l3_local = Path(args.l3)
    if args.no_l3:
        l3_ref = None
    elif l3_local.exists():
        wfiles = sorted(l3_local.joinpath(args.l3_subfolder).glob("*.safetensors")) or \
                 sorted(l3_local.joinpath(args.l3_subfolder).glob("*.bin"))
        hashes["l3_weights"] = {w.name: sha256_file(w) for w in wfiles}
        l3_ref = {"path": str(l3_local), "subfolder": args.l3_subfolder}
    else:
        l3_ref = {"hub_repo": args.l3, "hub_sha": hub_sha(args.l3), "subfolder": args.l3_subfolder}

    cascade_rows, ablation, repeat_report, borderline = [], {}, {}, {}
    for name, df in data.items():
        texts = df["text"].astype(str).tolist()
        y = df["label"].astype(int).to_numpy()
        scored_texts = [prenorm(t, args.prenorm) for t in texts]
        runs = [score_texts(scored_texts, models) for _ in range(args.repeat)]
        sc = runs[0]
        if args.repeat > 1:
            repeat_report[name] = {
                "max_abs_diff_l2": float(max(np.abs(r["s2"] - sc["s2"]).max() for r in runs[1:])),
                "max_abs_diff_l3": float(max(np.abs(r["p3"] - sc["p3"]).max() for r in runs[1:])),
                "decisions_identical": all(
                    np.array_equal(derive(r)[1], derive(sc)[1]) for r in runs[1:]
                ),
            }
        combos, first = derive(sc)
        cascade_rows.append(cascade_row(name, y, first))
        ablation[name] = {c: metrics(y, combos[c]) for c in COMBOS}
        borderline[name] = {
            "l2_within_0.01_of_threshold": int((np.abs(sc["s2"] - L2_THRESHOLD) < 0.01).sum()),
            "l3_within_0.05_of_threshold": int((np.abs(sc["p3"] - L3_THRESHOLD) < 0.05).sum()),
        }
        pd.DataFrame({
            "text_sha1": [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts],
            "label": y, "l1_blocked": sc["b1"], "l2_score": sc["s2"], "l3_prob": sc["p3"],
            "first_blocking_layer": first,
        }).to_csv(run_dir / f"decisions_{name}.csv", index=False)

    pd.DataFrame(cascade_rows).to_csv(run_dir / "cascade_summary.csv", index=False)
    result = {
        "meta": {
            "tag": args.tag, "utc": stamp, "seed": args.seed, "prenorm": args.prenorm, "device": models["device"],
            "thresholds": {"l2": L2_THRESHOLD, "l3": L3_THRESHOLD},
            "l3": l3_ref, "hashes": hashes, "git": git_info(), "libs": lib_versions(),
            "runtime_s": round(time.time() - t0, 1),
        },
        "cascade": cascade_rows,
        "ablation": ablation,
        "borderline": borderline,
        "repeat": repeat_report,
    }
    if args.expect:
        result["expect"] = compare_expected(args.expect, cascade_rows)
    (run_dir / "results.json").write_text(json.dumps(result, indent=2, default=float))

    # ---- console summary
    print(f"\nrun dir: {run_dir}\n")
    hdr = f"{'set':<12}{'n':>5}{'FPR':>9}{'FPR 95% CI':>18}{'det':>9}{'det 95% CI':>18}{'F1':>8}   by L1/L2/L3"
    print(hdr)
    for r in cascade_rows:
        fpr = f"{r['fpr']:.3f}" if "fpr" in r else "  -"
        fci = f"[{r['fpr_ci_lo']:.3f},{r['fpr_ci_hi']:.3f}]" if "fpr" in r else ""
        det = f"{r['detection_rate']:.3f}" if "detection_rate" in r else "  -"
        dci = f"[{r['det_ci_lo']:.3f},{r['det_ci_hi']:.3f}]" if "detection_rate" in r else ""
        f1 = f"{r['f1']:.3f}" if "f1" in r else "  -"
        print(f"{r['test_set']:<12}{r['n']:>5}{fpr:>9}{fci:>18}{det:>9}{dci:>18}{f1:>8}   "
              f"{r['blocked_by_l1']}/{r['blocked_by_l2']}/{r['blocked_by_l3']}")
    if repeat_report:
        print("\ndeterminism:", json.dumps(repeat_report))
    print("\nborderline scores:", json.dumps(borderline))
    if args.expect:
        print("\nvs expected:", json.dumps(result["expect"], indent=2))


if __name__ == "__main__":
    main()
