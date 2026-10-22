#!/usr/bin/env python3
"""Final held-out summary from saved per-sample decisions (no models needed).

Reads results/decisions/<cond>_s<seed>__<mode>/decisions_{standard,evasion,adversarial,ood}.csv written by
run_sweep.py --save-decisions, recomputes every rule offline, and reports mean +- std over seeds.

Rules (L1 = regex block, L2 = MiniLM/index score, L3 = classifier probability):
  cascade @0.50      L1 or L2 > 0.50 or L3 > 0.50      (the original design)
  cascade @t*        L1 or L2 > t*  or L3 > 0.50       (t* chosen on the DEV half only, see calibrate_l2.py)
  L2 off             L1 or L3 > 0.50
  L3 alone           L3 > 0.50

Metrics: std_det / std_fpr (test.csv), evas_det (test_evasion.csv), adv_all_fpr, adv_clean_fpr (held-out
adversarial items confirmed benign by the audit), ood_fpr.

    python evaluation/scripts/summarize_heldout.py --decisions-dir <results>/decisions \
        --assignments data/split_assignments.csv --l2-threshold 0.65
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["std_det", "std_fpr", "evas_det", "adv_all_fpr", "adv_clean_fpr", "ood_fpr"]


def rule_flags(df, rule, t_star):
    l1, l2, l3 = df["l1_blocked"].astype(bool), df["l2_score"], df["l3_prob"]
    return {
        "cascade @0.50": l1 | (l2 > 0.50) | (l3 > 0.50),
        "cascade @t*": l1 | (l2 > t_star) | (l3 > 0.50),
        "L2 off": l1 | (l3 > 0.50),
        "L3 alone": l3 > 0.50,
    }[rule]


RULES = ["cascade @0.50", "cascade @t*", "L2 off", "L3 alone"]


def run_metrics(folder, status, t_star):
    d = {n: pd.read_csv(folder / f"decisions_{n}.csv") for n in ["standard", "evasion", "adversarial", "ood"]}
    adv_status = d["adversarial"]["text_sha1"].map(status).fillna("unaudited")
    out = {}
    for rule in RULES:
        f = {n: rule_flags(df, rule, t_star).to_numpy() for n, df in d.items()}
        pos = d["standard"]["label"].to_numpy() == 1
        clean = (adv_status == "clean").to_numpy()
        out[rule] = {
            "std_det": float(f["standard"][pos].mean()), "std_fpr": float(f["standard"][~pos].mean()),
            "evas_det": float(f["evasion"].mean()), "adv_all_fpr": float(f["adversarial"].mean()),
            "adv_clean_fpr": float(f["adversarial"][clean].mean()) if clean.any() else float("nan"),
            "ood_fpr": float(f["ood"].mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions-dir", required=True)
    ap.add_argument("--assignments", required=True, help="data/split_assignments.csv from make_dev_split.py")
    ap.add_argument("--l2-threshold", type=float, required=True, help="t* chosen on dev")
    a = ap.parse_args()

    asg = pd.read_csv(a.assignments)
    asg = asg[asg["set"] == "adversarial"]
    status = dict(zip(asg.text_sha1, asg.audit_status))
    rows = []
    for folder in sorted(Path(a.decisions_dir).iterdir()):
        m = re.match(r"(?P<cond>.+)_s(?P<seed>\d+)__(?P<mode>\w+)$", folder.name)
        if not m or not (folder / "decisions_standard.csv").exists():
            continue
        for rule, vals in run_metrics(folder, status, a.l2_threshold).items():
            rows.append({"cond": m["cond"], "seed": int(m["seed"]), "mode": m["mode"], "rule": rule, **vals})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no decision folders found")
        return
    for mode in sorted(df["mode"].unique()):
        for cond in sorted(df["cond"].unique()):
            sub = df[(df["mode"] == mode) & (df["cond"] == cond)]
            print(f"\n== {cond} | normalization={mode} | seeds {sorted(sub.seed.unique())} | t*={a.l2_threshold} (mean +- std) ==")
            print(f"{'rule':<16}" + "".join(f"{c:>16}" for c in COLS))
            for rule in RULES:
                s = sub[sub.rule == rule]
                cells = []
                for c in COLS:
                    v = s[c].to_numpy(dtype=float)
                    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                    cells.append(f"{np.nanmean(v):.3f}+-{sd:.3f}")
                print(f"{rule:<16}" + "".join(f"{x:>16}" for x in cells))
    out = Path(a.decisions_dir).parent / "heldout_summary.csv"
    df.to_csv(out, index=False)
    print("\nper-run table written to", out)


if __name__ == "__main__":
    main()
