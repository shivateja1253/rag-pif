import re
import unicodedata

INJECTION_PATTERNS = [
    r"ignore\s+(\w+\s+)*(instructions?|prompt|context|rules?)",
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

def normalize_text(text):
    return unicodedata.normalize("NFKC", text).lower()

def layer1_filter(text: str) -> dict:
    normalized = normalize_text(text)
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized):
            return {"blocked": True, "layer": 1, "confidence": 1.0}
    return {"blocked": False, "layer": 1, "confidence": 0.0}
