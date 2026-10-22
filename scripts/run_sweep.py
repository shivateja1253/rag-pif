#!/usr/bin/env python3
"""Resumable sweep: train L3 under several hard-negative conditions x several seeds,
benchmark each on the clean v2.1 test sets, and summarize mean +/- std.

Conditions (extra benign rows added to data/v2_1/train.csv, taken from data/train_augmented.csv):
  C0_none  no extra rows
  C3_both  the 36 'adversarial' + 40 'synthetic_enterprise' rows (the original recipe)
  C1_adv   only the 36 'adversarial' rows
  C2_ent   only the 40 'synthetic_enterprise' rows

Each finished run writes one small JSON to --results-dir (put it on Drive). Re-running the
same command skips finished runs, so a Colab reset costs at most one run. Models are deleted
after benchmarking to save disk.

    python scripts/run_sweep.py --seeds 42,1,2,3 --max-runs 4     # repeat until it says complete
    python scripts/run_sweep.py --summarize                       # print the table only
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

COND_SOURCES = {
    "C0_none": None,
    "C3_both": "adversarial,synthetic_enterprise",
    "C1_adv": "adversarial",
    "C2_ent": "synthetic_enterprise",
}
COMBOS = ["L1", "L2", "L3", "L1+L2", "L1+L3", "L2+L3", "L1+L2+L3"]
COLS = [("std_det", "std det"), ("std_fpr", "std FPR"), ("evas_det", "evas det"),
        ("adv_fpr", "adv FPR"), ("ood_fpr", "ood FPR")]


def plan(seeds, conds=None):
    """Paired C0/C3 first for every seed, then the single-source conditions."""
    runs = [(c, s) for s in seeds for c in ("C0_none", "C3_both")]
    runs += [(c, s) for s in seeds for c in ("C1_adv", "C2_ent")]
    return [r for r in runs if conds is None or r[0] in conds]


def compact(results):
    a = results["ablation"]
    return {c: {
        "std_det": a["standard"][c].get("detection_rate"),
        "std_fpr": a["standard"][c].get("fpr"),
        "evas_det": a["evasion"][c].get("detection_rate"),
        "adv_fpr": a["adversarial"][c].get("fpr"),
        "ood_fpr": a["ood"][c].get("fpr"),
    } for c in COMBOS}


def run_one(cond, seed, a):
    tag = f"{cond}_s{seed}"
    res_path = Path(a.results_dir) / f"{tag}.json"
    if res_path.exists():
        return "skipped"
    eval_dir = getattr(a, "eval_dir", None) or a.data_dir
    modes = [m for m in getattr(a, "prenorm_modes", "none").split(",") if m]
    model_root = Path(a.work_dir) / tag
    ckpt = f"/tmp/ckpt_{tag}"
    cmd = [sys.executable, "scripts/train_l3.py", "--train", f"{a.data_dir}/train.csv",
           "--val", f"{a.data_dir}/val.csv", "--out", str(model_root), "--seed", str(seed), "--ckpt-dir", ckpt,
           "--select", a.select]
    if COND_SOURCES[cond]:
        cmd += ["--extra", a.extra_file, "--extra-sources", COND_SOURCES[cond]]
    subprocess.run(cmd, check=True)
    first = None
    for mode in modes:
        btag = f"{tag}__{mode}"
        bench = [sys.executable, "evaluation/scripts/run_benchmark.py", "--tag", btag, "--data-dir", eval_dir,
                 "--faiss", a.faiss, "--l3", str(model_root / "final"), "--device", a.device,
                 "--out-dir", str(Path(a.work_dir) / "runs"), "--seed", str(seed), "--prenorm", mode]
        subprocess.run(bench, check=True)
        rj = sorted((Path(a.work_dir) / "runs").glob(f"{btag}_*/results.json"))[-1]
        if getattr(a, "save_decisions", False):
            dst = Path(a.results_dir) / "decisions" / btag
            dst.mkdir(parents=True, exist_ok=True)
            for f in rj.parent.glob("decisions_*.csv"):
                shutil.copy(f, dst / f.name)
        if first is None:
            r = json.loads(rj.read_text())
            first = (r["cascade"], compact(r))
    meta = json.loads((model_root / "train_meta.json").read_text())
    payload = {"cond": cond, "seed": seed, "train_rows": meta["train_rows"],
               "best_val_f1": meta["best_val"].get("eval_f1"), "selection": meta.get("selection"),
               "eval_history": meta.get("eval_history"), "prenorm_modes": modes, "eval_dir": eval_dir,
               "metrics": first[1], "cascade": first[0]}
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(payload, indent=2))
    shutil.rmtree(model_root, ignore_errors=True)
    shutil.rmtree(ckpt, ignore_errors=True)
    return "done"


def summarize(results_dir):
    rows = [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("*.json"))]
    if not rows:
        print("no finished runs yet")
        return
    by = {}
    for r in rows:
        by.setdefault(r["cond"], []).append(r)

    def ms(vals):
        v = [x for x in vals if x is not None]
        if not v:
            return "      -      "
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        return f"{np.mean(v):.3f}+-{sd:.3f}"

    for layer in ["L3", "L1+L2+L3", "L1+L2"]:
        print(f"\n== {layer} (mean +- std over seeds) ==")
        print(f"{'condition':<10}{'seeds':>6}" + "".join(f"{lab:>14}" for _, lab in COLS))
        for cond in COND_SOURCES:
            if cond not in by:
                continue
            rs = by[cond]
            print(f"{cond:<10}{len(rs):>6}" + "".join(
                f"{ms([r['metrics'][layer][k] for r in rs]):>14}" for k, _ in COLS))
    print("\nseeds per condition:", {c: sorted(r["seed"] for r in rs) for c, rs in by.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="42,1,2,3")
    ap.add_argument("--max-runs", type=int, default=0, help="stop after this many NEW runs (0 = all)")
    ap.add_argument("--data-dir", default="data/v2_1")
    ap.add_argument("--extra-file", default="data/train_augmented.csv")
    ap.add_argument("--faiss", default="datasets/injection_index_v2_1.faiss")
    ap.add_argument("--results-dir", default="/content/drive/MyDrive/rag-pif/sweep_last")
    ap.add_argument("--select", default="last", choices=["last", "val_loss", "val_f1"])
    ap.add_argument("--eval-dir", default=None, help="folder with the test sets to benchmark on (default: --data-dir)")
    ap.add_argument("--conds", default=None, help="comma list, e.g. C0_none,C3_both (default: all four)")
    ap.add_argument("--prenorm-modes", default="none", help="comma list of none,full: benchmark once per mode")
    ap.add_argument("--save-decisions", action="store_true", help="copy per-sample decision CSVs into results-dir/decisions")
    ap.add_argument("--work-dir", default="/content/sweep_work")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()

    if not a.summarize:
        seeds = [int(x) for x in a.seeds.split(",")]
        conds = [c.strip() for c in a.conds.split(",")] if a.conds else None
        todo = plan(seeds, conds)
        done_now = 0
        for cond, seed in todo:
            if a.max_runs and done_now >= a.max_runs:
                break
            print(f"\n######## {cond} seed {seed} ########", flush=True)
            if run_one(cond, seed, a) == "done":
                done_now += 1
        remaining = [(c, s) for c, s in todo if not (Path(a.results_dir) / f"{c}_s{s}.json").exists()]
        print(f"\nnew runs this call: {done_now} | remaining: {len(remaining)}")
        print("ALL RUNS COMPLETE" if not remaining else "rerun the same command to continue")
    summarize(a.results_dir)


if __name__ == "__main__":
    main()
