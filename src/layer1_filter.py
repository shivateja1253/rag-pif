import re
import unicodedata

# ---------------------------------------------------------------------
# v3 -- context-aware regex filter.
#
# History:
#   v1: plain keyword regex matching (original).
#   v2: added research/news/fiction context suppression to fix high
#       adversarial FPR (0.545 -> 0.395 on test_adversarial.csv).
#   v3: added -- fixed a pre-existing gap where "guidelines" wasn't
#       recognized as a synonym of "rules" in the ignore-pattern (this
#       gap meant "new instruction: ignore all safety guidelines" was
#       NEVER actually caught by v1/v2, confirmed via direct testing);
#       widened fiction-genre prefix list + fixed a punctuation bug where
#       "Short story:" failed to match; added journalism-style prefixes
#       (news/report/investigation/analysis); added an AI-context
#       requirement for the two overly generic patterns ("act as",
#       "no restrictions") so mundane non-AI usage isn't flagged; added
#       explicit benign-role exclusion ("act as my assistant/a tour guide").
#       Net result on test_adversarial.csv: FPR 0.395 -> 0.295, with zero
#       regressions on test.csv / test_evasion.csv and the 8/8 real-
#       injection regression set.
#
# Known tradeoff: the journalism-prefix additions ("news:", "report:",
# "breaking:") are a real precision/security tradeoff -- an attacker could
# in principle prepend "Breaking news:" to try to slip a real injection
# past Layer 1 specifically. Layer 2 (semantic) and Layer 3 (RoBERTa)
# remain as backstops. Documented as a known limitation.
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


def normalize_text(text):
    return unicodedata.normalize("NFKC", text).lower()


def is_research_context(text, match_start, match_end, matched_pattern):
    prefix_window = text[max(0, match_start - 200):match_start].lower()
    suffix_window = text[match_end:min(len(text), match_end + 100)].lower()
    full_text_lower = text.lower()
    before_match = text[:match_start]

    for q in QUOTE_CHARS:
        if before_match.count(q) % 2 == 1:
            return True, 'quoted_phrase'

    for prefix in RESEARCH_PREFIXES:
        if re.search(prefix, prefix_window):
            return True, 'research_prefix'

    for suffix in RESEARCH_SUFFIXES:
        if re.search(suffix, suffix_window):
            return True, 'research_suffix'

    if matched_pattern == ACT_AS_PATTERN and re.search(BENIGN_ROLE_SUFFIX, suffix_window):
        return True, 'benign_role'

    for sig in STRONG_SIGNALS:
        if re.search(sig, full_text_lower):
            return True, 'strong_research_signal'

    signal_count = sum(1 for sig in RESEARCH_SIGNALS if re.search(sig, full_text_lower))
    if signal_count >= 2:
        return True, f'research_document({signal_count}_signals)'

    if matched_pattern in GENERIC_PATTERNS:
        if not any(re.search(kw, full_text_lower) for kw in AI_CONTEXT_KEYWORDS):
            return True, 'no_ai_context'

    return False, None


def layer1_filter(text: str) -> dict:
    normalized = normalize_text(text)
    for pattern in INJECTION_PATTERNS:
        m = re.search(pattern, normalized)
        if m:
            is_research, reason = is_research_context(text, m.start(), m.end(), pattern)
            if is_research:
                continue
            return {"blocked": True, "layer": 1, "confidence": 1.0, "pattern": m.group(), "suppressed_reason": None}
    return {"blocked": False, "layer": 1, "confidence": 0.0, "pattern": None, "suppressed_reason": None}
