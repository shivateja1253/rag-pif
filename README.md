# RAG-PIF: Prompt Injection Firewall for RAG Systems

A multi-layer prompt injection firewall for Retrieval-Augmented Generation systems.

## Repository structure
```
rag-pif/
├── streamlit/              # Demo application (3-layer firewall integrated)
│   ├── Home.py
│   ├── layer1_filter.py    # Layer 1: context-aware regex (v3)
│   ├── layer2_embedding.py # Layer 2: FAISS embedding similarity
│   ├── rag_firewall.py     # Full L1->L2->L3 cascade module
│   └── pages/
│       ├── 1_Chatbot.py
│       └── 2_Dashboard.py
├── models/                 # Layer 3 checkpoint (roberta_best_v2.pt), tokenizer, HF model dir
├── datasets/                # Training/val/test data, FAISS index
├── evaluation/              # Benchmark results (CSVs), including FINAL_benchmark_results.csv
├── paper/                   # IEEE Access paper draft
└── figures/                 # Paper figures/charts
```

## Architecture
- **Layer 1**: Context-aware regex filter (v3) -- fast keyword/pattern matching with
  research/news/fiction context suppression to reduce false positives on benign
  security-discussion text.
- **Layer 2**: Sentence-transformer (`all-MiniLM-L6-v2`) + FAISS embedding classifier,
  threshold 0.50.
- **Layer 3**: Fine-tuned **RoBERTa-base** classifier (switched from DeBERTa-v3 during
  Month 3 due to NaN-gradient issues on Colab's CUDA environment), threshold 0.50.

All three layers cascade (L1 -> L2 -> L3, early exit on first block) via
`streamlit/rag_firewall.py`, and are integrated end-to-end in the Streamlit demo.

## Running the demo
```
pip install -r requirements.txt
streamlit run streamlit/Home.py
```

## Final evaluation (Month 3, full pipeline, held-out test sets)

| Test set | F1 | FPR | Detection rate | n |
|---|---|---|---|---|
| Standard | 0.996 | 0.008 | 1.00 | 263 |
| Evasion (obfuscated/unicode/zero-width) | -- | 0.000 | 1.00 | 150 |
| Adversarial (benign security-research text) | -- | 0.280 | -- | 200 |
| OOD (enterprise emails/support tickets) | -- | 0.053 | -- | 170 |

Full per-layer breakdown and reproducibility metadata (checkpoint timestamps, test-file
hashes) in `evaluation/FINAL_benchmark_results.csv`.

### Known limitations
- Layer 1's journalism-style context prefixes ("news:", "report:", "breaking:") are a
  real precision/security tradeoff -- an attacker could in principle prepend such a
  prefix to try to slip a real injection past Layer 1 specifically. Layers 2 and 3
  remain as backstops.
- Layer 3 generalizes well to character-substitution evasion (e.g. leetspeak) but
  inconsistently to informal/abbreviated semantic paraphrasing with no recognizable
  trigger words (e.g. "kindly disregrd earlier guidance").
- A targeted hard-negative retraining experiment (`roberta_best_v3.pt`, not used in
  production) improved adversarial FPR further but regressed OOD FPR substantially
  (0.053 -> 0.141), illustrating that Layer 3's remaining false positives reflect a
  genuine precision/generalization tradeoff rather than a simple coverage gap. The
  verified `roberta_best_v2.pt` checkpoint was kept as production instead.

## Target: IEEE Access
