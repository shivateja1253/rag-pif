#!/usr/bin/env python3
"""Score published prompt-injection detectors on the same clean v2.1 test sets.

Detectors (each is skipped with a printed reason if it cannot load):
  protectai_deberta_v2   protectai/deberta-v3-base-prompt-injection-v2
  piguard                leolee99/PIGuard  (custom code; its author reports it breaks on newer
                         Python/transformers, so a load failure here is expected and is reported)
  prompt_guard_2_86m     meta-llama/Llama-Prompt-Guard-2-86M  (gated: accept the license on
                         Hugging Face and set HF_TOKEN, otherwise it is skipped)

Every detector is scored on the raw text with truncation at 512 tokens and P(injection) taken
from the softmax. The injection class is found from config.id2label and printed, so check it.
Results are reported at P(injection) >= 0.5 and >= 0.9. The 0.9 column is a sensitivity check,
not a tuned operating point.

    python evaluation/scripts/run_baselines.py --data-dir data/v2_1
    python evaluation/scripts/run_baselines.py --detectors protectai_deberta_v2
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_benchmark as rb  # noqa: E402  reuse metrics(), SETS, find_file
from textnorm import prenorm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
THRESHOLDS = [0.5, 0.9]
DETECTORS = {
    "protectai_deberta_v2": {"repo": "protectai/deberta-v3-base-prompt-injection-v2",
                             "tokenizer": "protectai/deberta-v3-base-prompt-injection-v2", "kwargs": {}},
    "piguard": {"repo": "leolee99/PIGuard", "tokenizer": "microsoft/deberta-v3-base",
                "kwargs": {"trust_remote_code": True}},
    "prompt_guard_2_86m": {"repo": "meta-llama/Llama-Prompt-Guard-2-86M",
                           "tokenizer": "meta-llama/Llama-Prompt-Guard-2-86M", "kwargs": {}},
}


def injection_index(id2label):
    """Index of the injection class, from the model's own label names."""
    names = {int(k): str(v) for k, v in id2label.items()}
    for i, n in names.items():
        if any(k in n.lower() for k in ("inject", "malic", "unsafe", "attack", "jailbreak")):
            return i
    if sorted(names) == [0, 1] and all(n.upper().startswith("LABEL_") for n in names.values()):
        return 1
    raise ValueError(f"cannot tell which label means injection: {names}")


def evaluate_probs(y, p, thr):
    """Same metrics as the main benchmark, from P(injection) at a threshold."""
    return rb.metrics(y, np.asarray(p) >= thr)


def score(texts, tok, model, idx, device, bs=32, max_len=512):
    import torch
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", truncation=True, max_length=max_len, padding=True)
        enc = {k: v.to(device) for k, v in enc.items() if k != "token_type_ids"}
        with torch.no_grad():
            logits = model(**enc).logits.float().cpu()
        out.append(torch.softmax(logits, dim=-1)[:, idx].numpy())
    return np.concatenate(out)


def headline(m_by_set):
    """(std det, std FPR, evasion det, adversarial FPR, OOD FPR) from per-set metric dicts."""
    g = lambda s, k: m_by_set[s].get(k)
    return (g("standard", "detection_rate"), g("standard", "fpr"), g("evasion", "detection_rate"),
            g("adversarial", "fpr"), g("ood", "fpr"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data" / "v2_1"))
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--out-dir", default=str(REPO / "evaluation" / "runs"))
    ap.add_argument("--prenorm", choices=["none", "full"], default="none",
                    help="normalization front-end applied before scoring (scripts/textnorm.py)")
    a = ap.parse_args()

    import gc
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {n: pd.read_csv(rb.find_file(a.data_dir, f)) for n, f in rb.SETS.items()}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(a.out_dir) / f"baselines_{a.prenorm}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"device {device} | transformers {transformers.__version__} | sets "
          f"{ {n: len(d) for n, d in data.items()} }")

    results, status = {}, {}
    for name in [d.strip() for d in a.detectors.split(",") if d.strip()]:
        spec = DETECTORS[name]
        try:
            tok = AutoTokenizer.from_pretrained(spec["tokenizer"])
            model = AutoModelForSequenceClassification.from_pretrained(spec["repo"], **spec["kwargs"]).to(device).eval()
            idx = injection_index(model.config.id2label)
            print(f"\n[{name}] id2label={dict(model.config.id2label)} -> injection index {idx}")
            probs = {n: score([prenorm(t, a.prenorm) for t in d["text"].astype(str).tolist()], tok, model, idx, device)
                     for n, d in data.items()}
            for n, d in data.items():
                pd.DataFrame({"text_sha1": [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in d["text"].astype(str)],
                              "label": d["label"].astype(int), "p_injection": probs[n]}
                             ).to_csv(run_dir / f"{name}_{n}.csv", index=False)
            results[name] = {thr: {n: evaluate_probs(d["label"].astype(int).to_numpy(), probs[n], thr)
                                   for n, d in data.items()} for thr in THRESHOLDS}
            status[name] = "ok"
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:  # report and continue with the next detector
            status[name] = f"SKIPPED: {type(e).__name__}: {str(e)[:300]}"
            print(f"\n[{name}] {status[name]}")

    (run_dir / "baselines.json").write_text(json.dumps({"status": status, "results": results}, indent=2, default=float))
    print("\n== baselines (rows: detector @ threshold) ==")
    print(f"{'detector':<24}{'thr':>5}{'std det':>9}{'std FPR':>9}{'evas det':>10}{'adv FPR':>9}{'ood FPR':>9}")
    f = lambda v: "    -" if v is None else f"{v:.3f}"
    for name, by_thr in results.items():
        for thr, m in by_thr.items():
            print(f"{name:<24}{thr:>5}" + "".join(f"{f(v):>{w}}" for v, w in zip(headline(m), (9, 9, 10, 9, 9))))
    print("\nstatus:", json.dumps(status, indent=1))
    print("wrote", run_dir)


if __name__ == "__main__":
    main()
