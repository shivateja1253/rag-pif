#!/usr/bin/env python3
"""Summarize the published detectors on the held-out half, next to our held-out table.

Reads the per-text CSVs written by run_baselines.py (baselines_<none|full>_<stamp>/<detector>_<set>.csv),
joins the audit status of the held-out adversarial items, and reports at P(injection) >= 0.5 and >= 0.9:
std_det, std_fpr, evas_det, adv_all_fpr, adv_clean_fpr (items confirmed benign by the audit), ood_fpr.

    python evaluation/scripts/summarize_baselines_heldout.py --runs-dir evaluation/runs --assignments data/split_assignments.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS = [0.5, 0.9]
COLS = ["std_det", "std_fpr", "evas_det", "adv_all_fpr", "adv_clean_fpr", "ood_fpr"]


def metrics(run, name, status, thr):
    d = {s: pd.read_csv(run / f"{name}_{s}.csv") for s in ["standard", "evasion", "adversarial", "ood"]}
    flag = {s: (df["p_injection"] >= thr).to_numpy() for s, df in d.items()}
    pos = d["standard"]["label"].to_numpy() == 1
    clean = (d["adversarial"]["text_sha1"].map(status).fillna("unaudited") == "clean").to_numpy()
    return {"std_det": float(flag["standard"][pos].mean()), "std_fpr": float(flag["standard"][~pos].mean()),
            "evas_det": float(flag["evasion"].mean()), "adv_all_fpr": float(flag["adversarial"].mean()),
            "adv_clean_fpr": float(flag["adversarial"][clean].mean()) if clean.any() else float("nan"),
            "ood_fpr": float(flag["ood"].mean())}


def newest_runs(runs_dir):
    best = {}
    for p in sorted(Path(runs_dir).glob("baselines_*_*")):
        m = re.match(r"baselines_(none|full)_(\d{8}T\d{6}Z)$", p.name)
        if m and m[1] not in best:
            best[m[1]] = p
        elif m:
            best[m[1]] = max(best[m[1]], p, key=lambda q: q.name)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="evaluation/runs")
    ap.add_argument("--assignments", required=True)
    a = ap.parse_args()
    asg = pd.read_csv(a.assignments)
    status = dict(zip(asg[asg["set"] == "adversarial"].text_sha1, asg[asg["set"] == "adversarial"].audit_status))
    rows = []
    for mode, run in newest_runs(a.runs_dir).items():
        names = sorted({re.match(r"(.+)_(standard|evasion|adversarial|ood)\.csv$", f.name)[1]
                        for f in run.glob("*_standard.csv")})
        for name in names:
            for thr in THRESHOLDS:
                rows.append({"mode": mode, "detector": name, "thr": thr, **metrics(run, name, status, thr)})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no baseline runs found")
        return
    for mode in sorted(df["mode"].unique()):
        print(f"\n== published detectors, held-out half, normalization={mode} ==")
        print(f"{'detector':<24}{'thr':>5}" + "".join(f"{c:>15}" for c in COLS))
        for _, r in df[df["mode"] == mode].iterrows():
            print(f"{r['detector']:<24}{r['thr']:>5}" + "".join(f"{r[c]:>15.3f}" for c in COLS))
    out = Path(a.runs_dir) / "baselines_heldout_summary.csv"
    df.to_csv(out, index=False)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
