#!/usr/bin/env python3
"""Split the hard-benign sets into DEV and HELD-OUT halves, and build two benchmark-ready folders.

Why: the L2 threshold (and any cascade rule) must be chosen on data that is never used for the final
report. The adversarial (200) and OOD (170) sets are each split 50/50, group-aware (near-duplicate texts
stay together) and stratified by audit status (confirmed benign / contested / not audited).

Outputs two folders that run_benchmark.py / run_baselines.py can read unchanged:
  <out>/dev_set      test.csv = val.csv, test_evasion.csv = val_evasion.csv, adversarial/OOD = dev halves
  <out>/heldout_set  test.csv = test.csv, test_evasion.csv = test_evasion.csv, adversarial/OOD = held-out halves
plus <out>/split_assignments.csv (text_sha1, set, half, audit_status). Held-out folders are only for the
final report: never tune anything on them.

    python scripts/make_dev_split.py --data-dir data/v2_1 --audit results/label_audit/audit_scored_all.csv --out data
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_splits import cluster  # noqa: E402


def sha(t):
    return hashlib.sha1(str(t).encode("utf-8")).hexdigest()


def audit_status(audit):
    """text_sha1 -> 'clean' | 'contested' for audited adversarial items."""
    a = audit[audit["set"] == "adversarial"] if "set" in audit.columns else audit
    clean = (a.directs_ai == "no") & (a.block_in_kb == "no")
    return dict(zip(a.text_sha1, np.where(clean, "clean", "contested")))


def split_halves(df, status, seed, thr):
    df = df.copy()
    df["text_sha1"] = df["text"].map(sha)
    df["status"] = df["text_sha1"].map(status).fillna("unaudited")
    df["group"] = cluster(df["text"].astype(str).tolist(), "word", thr)
    g = df.groupby("group").agg(size=("text", "size"), status=("status", lambda s: s.value_counts().index[0])).reset_index()
    rng = np.random.default_rng(seed)
    half = {}
    for st, gg in g.groupby("status"):
        order = gg.assign(r=rng.random(len(gg))).sort_values(["size", "r"], ascending=[False, True])
        have = {"dev": 0, "heldout": 0}
        for _, row in order.iterrows():
            pick = "dev" if have["dev"] <= have["heldout"] else "heldout"
            half[row["group"]] = pick
            have[pick] += row["size"]
    df["half"] = df["group"].map(half)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/v2_1")
    ap.add_argument("--audit", required=True, help="CSV with text_sha1, set, directs_ai, block_in_kb (parts 1 and 2)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-thr", type=float, default=0.5, help="word TF-IDF cosine for near-duplicate groups")
    a = ap.parse_args()

    d, out = Path(a.data_dir), Path(a.out)
    status = audit_status(pd.read_csv(a.audit))
    adv = split_halves(pd.read_csv(d / "test_adversarial.csv"), status, a.seed, a.group_thr)
    ood = split_halves(pd.read_csv(d / "test_ood.csv"), {}, a.seed + 1, a.group_thr)
    sets = {
        "dev_set": {"test.csv": d / "val.csv", "test_evasion.csv": d / "val_evasion.csv", "half": "dev"},
        "heldout_set": {"test.csv": d / "test.csv", "test_evasion.csv": d / "test_evasion.csv", "half": "heldout"},
    }
    for name, spec in sets.items():
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        for fname in ["test.csv", "test_evasion.csv"]:
            shutil.copy(spec[fname], folder / fname)
        cols = [c for c in pd.read_csv(d / "test_adversarial.csv").columns]
        adv[adv.half == spec["half"]][cols].to_csv(folder / "test_adversarial.csv", index=False)
        ood[ood.half == spec["half"]][cols].to_csv(folder / "test_ood.csv", index=False)

    rows = pd.concat([adv.assign(set="adversarial"), ood.assign(set="ood")], ignore_index=True)[["text_sha1", "set", "half", "status"]]
    rows.rename(columns={"status": "audit_status"}).to_csv(out / "split_assignments.csv", index=False)
    print(pd.crosstab([rows.set, rows.status], rows.half).to_string())
    bad = set(adv[adv.half == "dev"].group) & set(adv[adv.half == "heldout"].group)
    print("groups crossing halves:", len(bad))


if __name__ == "__main__":
    main()
