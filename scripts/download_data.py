#!/usr/bin/env python3
"""Download the RAG-PIF dataset (and optionally the model) from Hugging Face.

Writes the resolved commit hashes to <data-dir>/HF_REVISIONS.json so a run can
be tied to exact dataset/model versions.

    python scripts/download_data.py --data-dir data
    python scripts/download_data.py --data-dir data --model-dir models/hf_prod
    python scripts/download_data.py --data-revision <sha> --model-dir models/hf_prod --model-revision <sha>
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DATASET = "shivateja1253/rag-pif-datasets"
MODEL = "shivateja1253/rag-pif-roberta"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--data-revision", default=None, help="dataset commit sha to pin")
    ap.add_argument("--model-dir", default=None, help="also download the model here")
    ap.add_argument("--model-revision", default=None, help="model commit sha to pin")
    a = ap.parse_args()

    api = HfApi()
    revisions = {}

    ds_sha = a.data_revision or api.dataset_info(DATASET).sha
    snapshot_download(repo_id=DATASET, repo_type="dataset", revision=ds_sha, local_dir=a.data_dir)
    revisions["dataset"] = {"repo": DATASET, "sha": ds_sha}
    print(f"dataset {DATASET}@{ds_sha[:10]} -> {a.data_dir}")

    if a.model_dir:
        m_sha = a.model_revision or api.model_info(MODEL).sha
        snapshot_download(repo_id=MODEL, revision=m_sha, local_dir=a.model_dir)
        revisions["model"] = {"repo": MODEL, "sha": m_sha}
        print(f"model {MODEL}@{m_sha[:10]} -> {a.model_dir}")

    out = Path(a.data_dir) / "HF_REVISIONS.json"
    out.write_text(json.dumps(revisions, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
