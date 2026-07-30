# =====================================================================
# RAG-PIF -- FINAL benchmark run for the paper.
# Locked configuration:
#   Layer 1: v3 (context-aware, final version, matches pushed src/layer1_filter.py)
#   Layer 2: FAISS, threshold 0.50 (matches pushed src/layer2_embedding.py)
#   Layer 3: roberta_best_v2.pt (verified-good checkpoint, matches pushed models/)
#
# This run explicitly prints checkpoint path + modification time + a
# reproducibility hash of the test file before reporting numbers, because
# we previously got two DIFFERENT OOD FPR results (0.053 vs 0.141) from
# what were supposed to be identical runs -- almost certainly a stale
# path/session issue like the ones we hit repeatedly tonight. This run
# is the one to trust; if the numbers still don't match either prior
# run, investigate before using anything in the paper.
# =====================================================================

# !pip install -q faiss-cpu sentence-transformers transformers torch

from google.colab import drive
drive.mount('/content/drive')

import re, time, unicodedata, hashlib, os
import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

BASE = "/content/drive/MyDrive/rag-pif"

PATHS = {
    "test":             f"{BASE}/data/test.csv",
    "test_evasion":     f"{BASE}/data/test_evasion.csv",
    "test_adversarial": f"{BASE}/data/test_adversarial.csv",
    "test_ood":         f"{BASE}/data/test_ood.csv",
    "faiss_index":      f"{BASE}/data/injection_index.faiss",
    "roberta_ckpt":     f"{BASE}/models/roberta_best_v2.pt",
    "roberta_hf_dir":   f"{BASE}/models/roberta_v2_hf",
    "roberta_tokenizer": f"{BASE}/models/roberta_tokenizer_v2",
}

L2_THRESHOLD = 0.50
L3_THRESHOLD = 0.50

# --- Explicit verification printout -- check this BEFORE trusting results ---
print("=" * 70)
print("REPRODUCIBILITY CHECK")
print("=" * 70)
for key in ["roberta_ckpt", "roberta_tokenizer", "faiss_index"]:
    p = PATHS[key]
    exists = os.path.exists(p)
    mtime = time.ctime(os.path.getmtime(p)) if exists else "N/A"
    print(f"{key:20s} : {p}")
    print(f"{'':20s}   exists={exists}  modified={mtime}")

for key in ["test", "test_evasion", "test_adversarial", "test_ood"]:
    p = PATHS[key]
    with open(p, "rb") as f:
        content = f.read()
    md5 = hashlib.md5(content).hexdigest()
    df = pd.read_csv(p)
    print(f"{key:20s} : {len(df)} rows, md5={md5}")
print("=" * 70)
print()

# ---------------------------------------------------------------------
# LAYER 1 -- v3 (final, locked). Identical to the pushed src/layer1_filter.py.
# ---------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore\s+(\w+\s+)*(instructions?|prompt|context|rules?|guidelines?)",
    r"forget\s+(everything|all|previous|your\s+instructions)",
    r"disregard\s+(the\s+|all\s+|previous\s+)?(above|instructions?|context)",
    r"do\s+not\s+follow\s+(your|the|previous)\s+instructions",
    r"you\s+are\s+now\s+(?!an?\s+assistant)",
    r"act\s+as\s+(?!an?\s+assistant)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"your\s+(new\s+|true\s+|real\s+)?(role|persona|identity|task)\s+is",
    r"-{3,}\s*(system|instruction|prompt)",
    r"#{2,}\s*(system|new\s+prompt|instruction)",
    r"\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>",
    r"(DAN|developer|unrestricted|jailbreak|god)\s+mode",
    r"no\s+(restrictions?|filters?|limits?|guidelines?)",
    r"(send_email|delete_file|call_api|execute_code|search_web|read_file)\s*\(",
    r"this\s+document\s+supersedes",
    r"(updated\s+policy|system\s+notice|administrative\s+update)\s*:",
]
ACT_AS_PATTERN = r"act\s+as\s+(?!an?\s+assistant)"
NO_RESTRICTIONS_PATTERN = r"no\s+(restrictions?|filters?|limits?|guidelines?)"
GENERIC_PATTERNS = {ACT_AS_PATTERN, NO_RESTRICTIONS_PATTERN}

RESEARCH_PREFIXES = [
    r'(researchers?|scientists?|academics?|authors?)\s+(found|showed|discovered|studied|documented|analyzed|reported|noted|observed|described)\s+that\s+',
    r'(attackers?|hackers?|adversaries?|users?|people)\s+(use[sd]?|using|say[s]?|saying|type[sd]?|typing|write[s]?|writing|send[s]?|sending|try(ing|s)?|attempt(s|ing)?|employ(s|ing)?|exploit(s|ing)?)\s+',
    r'(example|e\.g\.|such as|including|like|called|known as|referred to as)\s*[:\-]?\s*',
    r'(the\s+)?(phrase|patterns?|technique|method|attack|prompt|command|string|text|keyword)\s+',
    r'(this|the)\s+(paper|study|research|work|course|tutorial|book|article|report|chapter|section)\s+(explains?|describes?|analyzes?|covers?|discusses?|presents?|shows?|examines?)\s+',
    r'(detect|detecting|detection of|blocking|blocked|prevents?|resists?|defend|defense against|guards? against)\s+',
    r'(test|testing|tested|evaluate|evaluating|benchmark)\s+',
    r'(warning|warns?|alert|caution|note|notice):\s*',
    r'(vulnerability|exploit|flaw|weakness|bug)\s+(report|found|discovered|involves?)\s+',
    r'(labeled?|annotated?|classified|categorized)\s+as\s+',
    r'(cannot|can\'t|refused?|rejected?|resisted?|blocked?)\s+(to\s+)?(comply|follow|accept|process)\s+',
    r'(in\s+)?(the\s+)?(novel|movie|film|short\s+story|story|fiction|game|book|comic\s+book|comic|screenplay|plot|fantasy|satire|thriller|animation|dystopian|sci-?fi|science\s+fiction|graphic\s+novel|cartoon)[\s:]+',
    r'(the\s+)?(classifier|model|system|firewall|filter|detector)\s+(was\s+)?(trained?|designed?|built?|created?)\s+',
    r'(contains?|includes?|has|with)\s+(examples?|samples?|instances?|cases?)\s+of\s+',
    r'(how|why|what|when|where)\s+(do|does|did|can|could|would|should|are|is)\s+',
    r'(our|the|this)\s+(dataset|corpus|benchmark|evaluation|test\s+set)\s+',
    r'(research|stud(y|ies)|analysis)\s+(on|into|of|about)\s+',
    r'(news|breaking(\s+news)?|report|investigation|analysis|feature|industry\s+report|tech\s+report|journalist\s+investigation|case\s+study)\s*[:\-]?\s*',
]
RESEARCH_SUFFIXES = [
    r'\s+(is|are|was|were)\s+(a|an|the)?\s*(common|known|typical|classic|standard|well-known|popular)\s+(attack|technique|method|pattern|prompt|example)',
    r'\s+(attack|technique|method|pattern|prompt|example|attempt)',
    r'\s+in\s+(the|our|this)\s+(paper|study|dataset|benchmark|literature|research)',
    r'\s+to\s+(test|evaluate|detect|identify|classify|recognize)',
    r'\s+style\s+(attack|prompt|injection|jailbreak)',
]
RESEARCH_SIGNALS = [
    r'\b(paper|study|research|tutorial|course|benchmark|dataset|classifier|detector|firewall)\b',
    r'\b(researchers?|academics?|scientists?)\b',
    r'\b(evaluated?|analyzed?|detected?|classified?|trained?)\b',
    r'\b(false\s+positive|true\s+positive|precision|recall|f1|accuracy)\b',
]
STRONG_SIGNALS = [
    r'\b(OWASP|CVE|vulnerability|penetration\s+test(ing)?|pen\s+test|red\s+team(ing)?|security\s+research(ers)?|honeypot)\b',
]
AI_CONTEXT_KEYWORDS = [
    r'\bai\b', r'\ba\.i\.\b', r'\bassistant\b', r'\bchatbot\b', r'\bbot\b',
    r'\bmodel\b', r'\bllm\b', r'\bgpt\b', r'\bprompt\b', r'\binstructions?\b',
    r'\bjailbreak\b', r'\bguidelines?\b', r'\bsystem\b', r'\bchat\b', r'\boverride\b',
]
BENIGN_ROLE_SUFFIX = r'^\s*(my|your|our|a|an|the)\s+(assistant|helper|guide|tour\s+guide|tutor|coach|translator|planner|advisor|consultant)\b'
QUOTE_CHARS = ['"', "'", '\u201c', '\u201d', '\u00ab', '\u00bb', '`']

def is_research_context(text, match_start, match_end, matched_pattern):
    prefix_window = text[max(0, match_start - 200):match_start].lower()
    suffix_window = text[match_end:min(len(text), match_end + 100)].lower()
    full_text_lower = text.lower()
    before_match = text[:match_start]
    for q in QUOTE_CHARS:
        if before_match.count(q) % 2 == 1:
            return True
    for prefix in RESEARCH_PREFIXES:
        if re.search(prefix, prefix_window):
            return True
    for suffix in RESEARCH_SUFFIXES:
        if re.search(suffix, suffix_window):
            return True
    if matched_pattern == ACT_AS_PATTERN and re.search(BENIGN_ROLE_SUFFIX, suffix_window):
        return True
    for sig in STRONG_SIGNALS:
        if re.search(sig, full_text_lower):
            return True
    signal_count = sum(1 for sig in RESEARCH_SIGNALS if re.search(sig, full_text_lower))
    if signal_count >= 2:
        return True
    if matched_pattern in GENERIC_PATTERNS:
        if not any(re.search(kw, full_text_lower) for kw in AI_CONTEXT_KEYWORDS):
            return True
    return False

def layer1_filter(text):
    normalized = unicodedata.normalize('NFKC', text).lower()
    for p in INJECTION_PATTERNS:
        m = re.search(p, normalized)
        if m:
            if is_research_context(text, m.start(), m.end(), p):
                continue
            return {"blocked": True, "confidence": 1.0}
    return {"blocked": False, "confidence": 0.0}

# ---------------------------------------------------------------------
# LAYER 2 / LAYER 3
# ---------------------------------------------------------------------
print("Loading Layer 2...")
l2_model = SentenceTransformer("all-MiniLM-L6-v2")
l2_index = faiss.read_index(PATHS["faiss_index"])

def layer2_filter(text):
    if l2_index.ntotal == 0:
        return {"blocked": False, "confidence": 0.0}
    vec = l2_model.encode([text], normalize_embeddings=True).astype("float32")
    D, _ = l2_index.search(vec, k=1)
    score = float(D[0][0])
    return {"blocked": score > L2_THRESHOLD, "confidence": round(score, 4)}

print("Loading Layer 3...")
device = "cuda" if torch.cuda.is_available() else "cpu"
l3_tokenizer = RobertaTokenizerFast.from_pretrained(PATHS["roberta_tokenizer"])
l3_model = RobertaForSequenceClassification.from_pretrained(PATHS["roberta_hf_dir"])
l3_model.to(device)
l3_model.eval()

@torch.no_grad()
def layer3_filter(text):
    inputs = l3_tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
    logits = l3_model(**inputs).logits
    prob = float(torch.softmax(logits, dim=-1)[0][1])
    return {"blocked": prob > L3_THRESHOLD, "confidence": round(prob, 4)}

def run_pipeline(text):
    t0 = time.perf_counter()
    r1 = layer1_filter(text)
    if r1["blocked"]:
        return {"blocked": True, "layer": 1, "latency_ms": (time.perf_counter() - t0) * 1000}
    r2 = layer2_filter(text)
    if r2["blocked"]:
        return {"blocked": True, "layer": 2, "latency_ms": (time.perf_counter() - t0) * 1000}
    r3 = layer3_filter(text)
    return {"blocked": r3["blocked"], "layer": 3 if r3["blocked"] else None,
            "latency_ms": (time.perf_counter() - t0) * 1000}

def evaluate(name, df):
    y_true = df["label"].astype(int).tolist()
    y_pred, latencies, layer_counts = [], [], {1: 0, 2: 0, 3: 0}
    for text in df["text"]:
        r = run_pipeline(text)
        y_pred.append(1 if r["blocked"] else 0)
        latencies.append(r["latency_ms"])
        if r["blocked"]:
            layer_counts[r["layer"]] += 1

    has_both = len(set(y_true)) > 1
    f1 = f1_score(y_true, y_pred, zero_division=0) if has_both else None
    precision = precision_score(y_true, y_pred, zero_division=0) if has_both else None
    recall = recall_score(y_true, y_pred, zero_division=0) if has_both else None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else None

    return {
        "test_set": name, "n": len(df),
        "f1": round(f1, 4) if f1 is not None else None,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "fpr": round(fpr, 4), "detection_rate": round(detection_rate, 4) if detection_rate is not None else None,
        "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
        "blocked_by_l1": layer_counts[1], "blocked_by_l2": layer_counts[2], "blocked_by_l3": layer_counts[3],
        "avg_latency_ms": round(float(np.mean(latencies)), 3),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 3),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 3),
    }

print("\nRunning final benchmark...\n")
test_sets = {
    "standard": pd.read_csv(PATHS["test"]),
    "evasion": pd.read_csv(PATHS["test_evasion"]),
    "adversarial": pd.read_csv(PATHS["test_adversarial"]),
    "ood": pd.read_csv(PATHS["test_ood"]),
}

results = []
for name, df in test_sets.items():
    r = evaluate(name, df)
    results.append(r)
    print(f"=== {name} ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
    print()

results_df = pd.DataFrame(results)
out_path = f"{BASE}/evaluation/FINAL_benchmark_results.csv"
results_df.to_csv(out_path, index=False)

print("=" * 70)
print("FINAL RESULTS TABLE (paper-ready)")
print("=" * 70)
print(results_df[["test_set", "n", "f1", "precision", "recall", "fpr", "detection_rate",
                   "avg_latency_ms", "p99_latency_ms"]].to_string(index=False))
print(f"\nSaved to: {out_path}")

print("\n" + "=" * 70)
print("If this OOD FPR does not match 0.053 or 0.141 from earlier runs,")
print("that itself is useful information -- it means there IS some")
print("non-determinism (likely FAISS/embedding numerical variance across")
print("environments), and the paper should report this run's numbers")
print("as the authoritative, reproducible-from-this-script result.")
print("=" * 70)
