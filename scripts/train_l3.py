#!/usr/bin/env python3
"""Train RAG-PIF Layer 3 (roberta-base) on the v2.1 splits with a VERIFIED tokenizer.

Why this exists
  The v1 checkpoints (production, v3, v4) were trained and are served with a tokenizer
  whose tokenizer.json has 0 BPE merge rules, so text is split into single characters
  (31 characters -> 33 tokens). This script fails loudly if that ever happens again:
    * the tokenizer must split a probe sentence into the exact stock roberta-base tokens
    * after saving, the tokenizer is reloaded from disk and must give identical input_ids
      to stock roberta-base on every validation text
    * the saved model is reloaded and re-scored; it must match the in-memory best model

Usage
  python scripts/train_l3.py --train data/v2_1/train.csv --val data/v2_1/val.csv --out models/l3_v2_1
  python scripts/train_l3.py ... --with-evasion      # also train on data/v2_1/train_evasion.csv
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic GPU matmuls

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

PROBE = "Paris is the capital of France."
PROBE_TOKENS = ["<s>", "Paris", "Ġis", "Ġthe", "Ġcapital", "Ġof", "ĠFrance", ".", "</s>"]


def assert_tokenizer_ok(tok, where):
    got = tok.convert_ids_to_tokens(tok(PROBE)["input_ids"])
    if got != PROBE_TOKENS:
        raise RuntimeError(f"Tokenizer from {where} is broken.\n  expected {PROBE_TOKENS}\n  got      {got}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_key(t):
    t = unicodedata.normalize("NFKC", str(t))
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")
    return re.sub(r"\s+", " ", t).strip().lower()


def assert_no_eval_overlap(extra_df, train_path):
    """Extra training rows must never match any evaluation text (exact, after normalization)."""
    d = Path(train_path).parent
    keys = set(extra_df["text"].map(norm_key))
    for name in ["val.csv", "test.csv", "test_evasion.csv", "test_adversarial.csv", "test_ood.csv"]:
        p = d / name
        if p.exists():
            hit = keys & set(pd.read_csv(p)["text"].astype(str).map(norm_key))
            if hit:
                raise RuntimeError(f"{len(hit)} extra training rows are identical to rows of {name}; refusing to train")


def load_df(df):
    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]]


def load(path):
    return load_df(pd.read_csv(path))


def f1_stats(preds, y):
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"f1": 2 * prec * rec / max(prec + rec, 1e-9), "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/v2_1/train.csv")
    ap.add_argument("--val", default="data/v2_1/val.csv")
    ap.add_argument("--out", default="models/l3_v2_1")
    ap.add_argument("--ckpt-dir", default="/tmp/l3_checkpoints", help="scratch dir for epoch checkpoints (keep off Drive)")
    ap.add_argument("--base", default="roberta-base")
    ap.add_argument("--with-evasion", action="store_true")
    ap.add_argument("--extra", default=None, help="CSV with extra training rows (columns text,label,source)")
    ap.add_argument("--extra-sources", default=None, help="comma list; keep only extra rows whose 'source' is in it")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1, help="fraction of total steps")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--select", choices=["last", "val_loss", "val_f1"], default="last",
                    help="which epoch to keep. 'last' is fully deterministic. val_f1 picks arbitrarily among "
                         "tied epochs when the val set is saturated (it is), so it is not the default.")
    ap.add_argument("--non-deterministic", action="store_true", help="skip torch deterministic mode")
    a = ap.parse_args()

    set_seed(a.seed)
    if not a.non_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(a.base)
    assert_tokenizer_ok(tok, a.base)
    print("tokenizer probe OK:", tok.convert_ids_to_tokens(tok(PROBE)["input_ids"]))

    train = load(a.train)
    if a.with_evasion:
        train = pd.concat([train, load(Path(a.train).with_name("train_evasion.csv"))], ignore_index=True)
    extra_info = None
    if a.extra:
        ex_raw = pd.read_csv(a.extra)
        if a.extra_sources:
            keep = {x.strip() for x in a.extra_sources.split(",")}
            ex_raw = ex_raw[ex_raw["source"].isin(keep)]
        ex = load_df(ex_raw)
        assert_no_eval_overlap(ex, a.train)
        train = pd.concat([train, ex], ignore_index=True)
        extra_info = {"file": a.extra, "sources": a.extra_sources, "rows": int(len(ex)),
                      "labels": {int(k): int(v) for k, v in ex["label"].value_counts().items()}}
        print(f"added {len(ex)} extra rows from {a.extra} (sources={a.extra_sources}), labels {extra_info['labels']}")
    val = load(a.val)
    counts = np.bincount(train["label"].to_numpy(), minlength=2)
    weights = torch.tensor(len(train) / (2.0 * counts), dtype=torch.float32)
    print(f"train rows {len(train)} (benign {counts[0]}, injection {counts[1]}) | val rows {len(val)} | "
          f"class weights {weights.numpy().round(3).tolist()}")

    def to_ds(df):
        ds = Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)
        ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_len),
                    batched=True, remove_columns=["text"])
        return ds.rename_column("label", "labels")

    train_ds, val_ds = to_ds(train), to_ds(val)

    model = AutoModelForSequenceClassification.from_pretrained(
        a.base, num_labels=2, id2label={0: "benign", 1: "injection"}, label2id={"benign": 0, "injection": 1})

    steps_per_epoch = math.ceil(len(train) / a.bs)
    targs = TrainingArguments(
        output_dir=a.ckpt_dir, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs, per_device_eval_batch_size=32,
        learning_rate=a.lr, weight_decay=a.wd,
        warmup_steps=int(a.warmup * steps_per_epoch * a.epochs),
        eval_strategy="epoch", save_strategy=("no" if a.select == "last" else "epoch"), save_total_limit=2,
        load_best_model_at_end=(a.select != "last"),
        metric_for_best_model={"last": None, "val_loss": "eval_loss", "val_f1": "eval_f1"}[a.select],
        greater_is_better={"last": None, "val_loss": False, "val_f1": True}[a.select],
        seed=a.seed, data_seed=a.seed, fp16=a.fp16, report_to="none", logging_steps=20)

    def compute_metrics(p):
        s = f1_stats(np.argmax(p.predictions, axis=-1), p.label_ids)
        return {k: float(s[k]) for k in ["f1", "precision", "recall"]}

    trainer = WeightedTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                              data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
                              class_weights=weights)
    print("\n=== TRAINING ===")
    trainer.train()
    best = trainer.evaluate()
    history = [{"epoch": e["epoch"], "eval_loss": e["eval_loss"], "eval_f1": e["eval_f1"]}
               for e in trainer.state.log_history if "eval_f1" in e]
    print("\nper-epoch val:", [(round(h["epoch"]), round(h["eval_loss"], 4), round(h["eval_f1"], 4)) for h in history])
    print(f"selection={a.select}; evaluated model on val:", {k: round(v, 4) for k, v in best.items() if k.startswith("eval_")})

    # ---- save, then verify the SAVED artifacts, not the in-memory ones
    final = out / "final"
    trainer.save_model(str(final))
    tok_how = "save_pretrained"
    tok.save_pretrained(str(final))
    try:
        assert_tokenizer_ok(AutoTokenizer.from_pretrained(str(final)), str(final))
    except RuntimeError as e:
        print("\nWARNING: tok.save_pretrained() produced a broken tokenizer folder:\n", e)
        print("Falling back to copying the stock roberta-base tokenizer files.")
        from huggingface_hub import snapshot_download
        snapshot_download(a.base, local_dir=str(final), allow_patterns=[
            "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"])
        tok_how = "copied from roberta-base (save_pretrained was broken)"
        assert_tokenizer_ok(AutoTokenizer.from_pretrained(str(final)), str(final))

    tok2 = AutoTokenizer.from_pretrained(str(final))
    mism = sum(tok2(t, truncation=True, max_length=a.max_len)["input_ids"] !=
               tok(t, truncation=True, max_length=a.max_len)["input_ids"] for t in val["text"])
    if mism:
        raise RuntimeError(f"saved tokenizer differs from {a.base} on {mism} val texts")
    print(f"saved tokenizer verified ({tok_how}); identical input_ids on all {len(val)} val texts")

    m2 = AutoModelForSequenceClassification.from_pretrained(str(final)).to(dev).eval()
    preds = []
    for i in range(0, len(val), 32):
        enc = tok2(val["text"].tolist()[i:i + 32], return_tensors="pt", truncation=True,
                   max_length=a.max_len, padding=True).to(dev)
        with torch.no_grad():
            preds.append(m2(**enc).logits.float().cpu())
    s = f1_stats(torch.cat(preds).argmax(-1).numpy(), val["label"].to_numpy())
    gap = abs(s["f1"] - best["eval_f1"])
    print(f"reloaded-from-disk val F1 {s['f1']:.4f} vs trainer best {best['eval_f1']:.4f} (gap {gap:.4f})")
    if gap > 0.02:
        raise RuntimeError("reloaded model does not match the in-memory best model")

    meta = {
        "utc": datetime.now(timezone.utc).isoformat(), "args": vars(a), "device": dev,
        "libs": {"python": platform.python_version(), "transformers": transformers.__version__,
                 "torch": torch.__version__},
        "data_sha256": {"train": sha256_file(a.train), "val": sha256_file(a.val)},
        "selection": a.select, "eval_history": history,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "train_rows": int(len(train)), "extra": extra_info, "class_weights": weights.tolist(),
        "best_val": {k: float(v) for k, v in best.items() if k.startswith("eval_")},
        "reload_val": s, "tokenizer_saved_via": tok_how,
        "weights_sha256": sha256_file(final / "model.safetensors"),
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print("\nDONE. model + tokenizer:", final)
    print("next: python evaluation/scripts/run_benchmark.py --tag l3_v2_1 --data-dir data/v2_1 "
          f"--faiss datasets/injection_index_v2_1.faiss --l3 {final} --device cpu")


if __name__ == "__main__":
    main()
