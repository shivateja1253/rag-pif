#!/usr/bin/env python3
"""Evasion control v2: paired detection / false-positive view of fullwidth, leetspeak and zero-width text.

For every system and every transform (plain, fullwidth, leetspeak, zero_width) it reports
  inj   share of INJECTION texts flagged   (plain = test.csv 'direct' injections; the others = test_evasion.csv)
  std   share of BENIGN texts flagged      (benign rows of test.csv, transformed)
  ood   share of BENIGN texts flagged      (test_ood.csv, transformed)
  bacc  balanced accuracy = 0.5 * (inj + 1 - mean(std, ood))   (0.5 means no separation)
and repeats everything after a normalization front-end applied to the text BEFORE any detector:
  none  raw text
  full  NFKC, U+200B -> space, other zero-width/format characters removed, and leet characters
        (0 1 3 4 5 7 @ $) mapped back to letters when at least a quarter of the tokens mix letters with them

If a detector flags transformed benign text as often as transformed injections, its "evasion
detection" says nothing about injections. Systems: L1, L2, each of our L3 models, the baselines.

    python evaluation/scripts/evasion_control.py --data-dir data/v2_1 \
        --faiss datasets/injection_index_v2_1.faiss \
        --models "no_extra=/content/drive/MyDrive/rag-pif/models/l3_v2_1/final,plus76=/content/drive/MyDrive/rag-pif/models/l3_v2_1_hn/final"
"""
import argparse
import gc
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))
import run_baselines as bl  # noqa: E402
import run_benchmark as rb  # noqa: E402
from build_splits import to_fullwidth, to_leet, to_zero_width  # noqa: E402

TRANSFORMS = {"plain": lambda t: t, "fullwidth": to_fullwidth, "leetspeak": to_leet, "zero_width": to_zero_width}
CATEGORY_TO_TRANSFORM = {"unicode": "fullwidth", "obfuscated": "leetspeak", "zero_width": "zero_width"}
MODES = ["none", "full"]
from textnorm import LEET_BACK, prenorm  # noqa: E402,F401


def build_texts(data_dir):
    std = pd.read_csv(rb.find_file(data_dir, "test.csv"))
    ood = pd.read_csv(rb.find_file(data_dir, "test_ood.csv"))
    ev = pd.read_csv(rb.find_file(data_dir, "test_evasion.csv"))
    benign = {"std": std[std.label == 0]["text"].astype(str).tolist(), "ood": ood["text"].astype(str).tolist()}
    texts = {f"{s}|{t}": [fn(x) for x in xs] for s, xs in benign.items() for t, fn in TRANSFORMS.items()}
    texts["inj|plain"] = std[(std.label == 1) & (std.category == "direct")]["text"].astype(str).tolist()
    for cat, t in CATEGORY_TO_TRANSFORM.items():
        texts[f"inj|{t}"] = ev[ev.category == cat]["text"].astype(str).tolist()
    return texts


def share(flags):
    return float(np.mean(np.asarray(flags, dtype=bool)))


def bacc(results, system, t):
    d, s, o = results[system].get(f"inj|{t}"), results[system].get(f"std|{t}"), results[system].get(f"ood|{t}")
    return None if None in (d, s, o) else 0.5 * (d + 1 - (s + o) / 2)


def render(results, mode):
    cols = list(TRANSFORMS)
    out = []
    systems = [s for s in results if results[s].get("mode") == mode]
    for title, key, what in [("INJECTIONS flagged (detection)", "inj", "higher is better"),
                             ("BENIGN standard flagged (false positives)", "std", "lower is better"),
                             ("BENIGN ood flagged (false positives)", "ood", "lower is better"),
                             ("BALANCED ACCURACY (0.5 = no separation)", "bacc", "higher is better")]:
        out.append(f"\n== normalization={mode} | {title} ({what}) ==")
        out.append(f"{'system':<40}" + "".join(f"{c:>12}" for c in cols))
        for s in systems:
            vals = [bacc(results, s, c) if key == "bacc" else results[s].get(f"{key}|{c}") for c in cols]
            out.append(f"{s.split(' | ')[0]:<40}" + "".join(f"{('-' if v is None else format(v, '.3f')):>12}" for v in vals))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data" / "v2_1"))
    ap.add_argument("--faiss", default=str(REPO / "datasets" / "injection_index_v2_1.faiss"))
    ap.add_argument("--models", required=True, help="comma list of name=path to our L3 model dirs")
    ap.add_argument("--baselines", default="protectai_deberta_v2,piguard", help="comma list, or 'none'")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=str(REPO / "evaluation" / "runs"))
    a = ap.parse_args()

    texts = build_texts(a.data_dir)
    print("texts per cell:", {k: len(v) for k, v in texts.items() if k.endswith("plain")})
    normed = {mode: {k: [prenorm(x, mode) for x in v] for k, v in texts.items()} for mode in MODES}
    results = {}

    def put(system, mode, by_key):
        results[f"{system} | {mode}"] = {"mode": mode, **by_key}

    first = True
    for item in [x for x in a.models.split(",") if x.strip()]:
        name, path = item.split("=", 1)
        m = rb.load_models(SimpleNamespace(device=a.device, seed=0, faiss=a.faiss, tokenizer=None,
                                           l3=path.strip(), l3_subfolder="", no_l3=False))
        for mode in MODES:
            sc = {k: rb.score_texts(t, m) for k, t in normed[mode].items()}
            if first:
                put("L1 regex", mode, {k: share(v["b1"]) for k, v in sc.items()})
                put("L2 (MiniLM + index)", mode, {k: share(v["s2"] > rb.L2_THRESHOLD) for k, v in sc.items()})
            put(f"ours L3 alone: {name}", mode, {k: share(v["p3"] > rb.L3_THRESHOLD) for k, v in sc.items()})
            put(f"ours cascade: {name}", mode, {k: share(v["b1"] | (v["s2"] > rb.L2_THRESHOLD) | (v["p3"] > rb.L3_THRESHOLD))
                                                for k, v in sc.items()})
        first = False
        del m
        gc.collect()

    status = {}
    if a.baselines.strip().lower() != "none":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for name in [x.strip() for x in a.baselines.split(",") if x.strip()]:
            spec = bl.DETECTORS[name]
            try:
                tok = AutoTokenizer.from_pretrained(spec["tokenizer"])
                model = AutoModelForSequenceClassification.from_pretrained(spec["repo"], **spec["kwargs"]).to(device).eval()
                idx = bl.injection_index(model.config.id2label)
                for mode in MODES:
                    put(f"{name} @0.5", mode, {k: share(bl.score(t, tok, model, idx, device) >= 0.5)
                                              for k, t in normed[mode].items()})
                status[name] = "ok"
                del model
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                status[name] = f"SKIPPED: {type(e).__name__}: {str(e)[:200]}"

    for mode in MODES:
        print(render(results, mode))
    if status:
        print("\nbaseline status:", json.dumps(status))
    out = Path(a.out_dir) / f"evasion_control_v2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "evasion_control_v2.json").write_text(json.dumps({"results": results, "status": status}, indent=2))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
