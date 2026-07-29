# RAG-PIF: Prompt Injection Firewall for RAG Systems

A multi-layer prompt injection firewall for Retrieval-Augmented Generation systems.

## Architecture
- **Layer 1**: Context-aware regex filter (v3) -- fast keyword/pattern matching with
  research/news/fiction context suppression to reduce false positives on benign
  security-discussion text.
- **Layer 2**: Sentence-transformer (`all-MiniLM-L6-v2`) + FAISS embedding classifier,
  threshold 0.50.
- **Layer 3**: Fine-tuned **RoBERTa-base** classifier (switched from DeBERTa-v3 during
  Month 3 due to NaN-gradient issues on Colab's CUDA environment), threshold 0.50.

All three layers cascade (L1 -> L2 -> L3, early exit on first block) via
`src/rag_firewall.py`, and are integrated end-to-end in the Streamlit demo
(`src/pages/1_Chatbot.py`).

## Evaluation (Month 3, full pipeline, held-out test sets)

| Test set | F1 | FPR | Detection rate | n |
|---|---|---|---|---|
| Standard | 0.996 | 0.008 | 1.00 | 263 |
| Evasion (obfuscated/unicode/zero-width) | -- | 0.00 | 1.00 | 150 |
| Adversarial (benign security-research text) | -- | 0.295 | -- | 200 |
| OOD (enterprise emails/support tickets) | -- | 0.053 | -- | 170 |

Full per-layer breakdown in `evaluation/full_pipeline_evaluation_v2.csv`.

### Known limitations
- Layer 1's journalism-style context prefixes ("news:", "report:", "breaking:") are a
  real precision/security tradeoff -- an attacker could in principle prepend such a
  prefix to try to slip a real injection past Layer 1 specifically. Layers 2 and 3
  remain as backstops.
- Layer 3 generalizes well to character-substitution evasion (e.g. leetspeak) but
  inconsistently to informal/abbreviated semantic paraphrasing with no recognizable
  trigger words (e.g. "kindly disregrd earlier guidance").
- A targeted hard-negative retraining experiment (`roberta_best_v3.pt`) improved
  adversarial FPR further (0.295 -> 0.265) but regressed OOD FPR substantially
  (0.053 -> 0.141). The production checkpoint (`roberta_best_v2.pt`) was kept instead,
  as the tradeoff was not favorable. This is documented as a negative result.

## Target: IEEE Access
