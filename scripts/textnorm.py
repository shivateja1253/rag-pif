"""Text normalization front-end shared by the benchmark and the evasion control."""
import re
import unicodedata

LEET_BACK = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def prenorm(text, mode):
    """mode 'none': unchanged. mode 'full': NFKC, U+200B -> space, other zero-width/format characters
    removed, and leetspeak characters mapped back to letters when at least a quarter of the tokens mix
    letters with them (so ordinary numbers like 2024 are left alone)."""
    if mode == "none":
        return text
    t = unicodedata.normalize("NFKC", text).replace("\u200b", " ")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")
    t = re.sub(r"[ \t]+", " ", t)
    toks = t.split()
    mixed = sum(1 for k in toks if re.search(r"[A-Za-z]", k) and re.search(r"[0-9@$]", k))
    if mixed >= 2 and mixed / max(len(toks), 1) >= 0.25:
        t = t.translate(LEET_BACK)
    return t
