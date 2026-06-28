"""
Persian Banking OCR — NLP Post-Processing (Phase 4)
Levenshtein correction + banking-specific regex filters
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Persian banking vocabulary
# ---------------------------------------------------------------------------

AMOUNT_WORDS = [
    "صفر","یک","دو","سه","چهار","پنج","شش","هفت","هشت","نه",
    "ده","یازده","دوازده","سیزده","چهارده","پانزده","شانزده","هفده","هجده","نوزده",
    "بیست","سی","چهل","پنجاه","شصت","هفتاد","هشتاد","نود",
    "صد","دویست","سیصد","چهارصد","پانصد","ششصد","هفتصد","هشتصد","نهصد",
    "هزار","میلیون","میلیارد","تریلیون","ریال","تومان","و",
]

PERSIAN_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩0123456789", "۰۱۲۳۴۵۶۷۸۹۰۱۲۳۴۵۶۷۸۹")
ARABIC_TO_FA      = str.maketrans("ي ك", "ی ک")


# ---------------------------------------------------------------------------
# Levenshtein (pure Python, no deps)
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1,
                            prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def closest_word(token: str, vocab: list, max_dist: int = 2) -> Optional[str]:
    best, best_d = None, max_dist + 1
    for w in vocab:
        d = levenshtein(token, w)
        if d < best_d:
            best, best_d = w, d
    return best if best_d <= max_dist else None


# ---------------------------------------------------------------------------
# Field-specific cleaners
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    text = text.translate(PERSIAN_DIGIT_MAP)
    text = text.translate(ARABIC_TO_FA)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_amount_numeric(raw: str) -> str:
    """Keep only Persian digits and separators."""
    raw = normalise(raw)
    cleaned = re.sub(r"[^۰-۹,،٬]", "", raw)
    return cleaned


def clean_amount_words(raw: str) -> str:
    """Correct each token against banking vocabulary."""
    raw = normalise(raw)
    tokens = raw.split()
    corrected = []
    for tok in tokens:
        if tok in AMOUNT_WORDS:
            corrected.append(tok)
        else:
            fix = closest_word(tok, AMOUNT_WORDS)
            corrected.append(fix if fix else tok)
    return " ".join(corrected)


def clean_account(raw: str) -> str:
    """Account numbers: digits and dashes only, remove stray letters."""
    raw = normalise(raw)
    # Keep only Persian digits, Western digits, and dashes
    cleaned = re.sub(r"[^۰-۹0-9\-]", "", raw)
    cleaned = cleaned.translate(PERSIAN_DIGIT_MAP)
    return cleaned


def clean_date(raw: str) -> str:
    """Ensure YYYY/MM/DD in Persian digits."""
    raw = normalise(raw)
    digits_only = re.sub(r"[^۰-۹/]", "", raw)
    return digits_only


def clean_payee(raw: str) -> str:
    """Remove non-Persian characters from name fields."""
    raw = normalise(raw)
    cleaned = re.sub(r"[^\u0600-\u06FF\s]", "", raw).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

CLEANERS = {
    "amount_numeric": clean_amount_numeric,
    "amount_words":   clean_amount_words,
    "account":        clean_account,
    "date":           clean_date,
    "payee":          clean_payee,
}


@dataclass
class PostProcessResult:
    field:     str
    raw:       str
    corrected: str
    changed:   bool


def postprocess_field(field: str, raw_text: str) -> PostProcessResult:
    cleaner   = CLEANERS.get(field, normalise)
    corrected = cleaner(raw_text)
    return PostProcessResult(
        field=field,
        raw=raw_text,
        corrected=corrected,
        changed=(corrected != raw_text),
    )


# ---------------------------------------------------------------------------
# Cross-validation: amount_numeric ↔ amount_words
# ---------------------------------------------------------------------------

WORD_TO_INT = {
    "صفر":0,"یک":1,"دو":2,"سه":3,"چهار":4,"پنج":5,"شش":6,"هفت":7,"هشت":8,"نه":9,
    "ده":10,"یازده":11,"دوازده":12,"سیزده":13,"چهارده":14,"پانزده":15,
    "شانزده":16,"هفده":17,"هجده":18,"نوزده":19,
    "بیست":20,"سی":30,"چهل":40,"پنجاه":50,"شصت":60,"هفتاد":70,"هشتاد":80,"نود":90,
    "صد":100,"دویست":200,"سیصد":300,"چهارصد":400,"پانصد":500,
    "ششصد":600,"هفتصد":700,"هشتصد":800,"نهصد":900,
    "هزار":1_000,"میلیون":1_000_000,"میلیارد":1_000_000_000,
}

def words_to_int(text: str) -> Optional[int]:
    """Best-effort conversion of Persian amount words to integer."""
    text = re.sub(r"(ریال|تومان)", "", text).strip()
    tokens = text.split()
    total, current = 0, 0
    for tok in tokens:
        if tok == "و":
            continue
        val = WORD_TO_INT.get(tok)
        if val is None:
            return None
        if val >= 1_000:
            current = max(current, 1) * val
            if val >= 1_000_000:
                total += current
                current = 0
        else:
            current += val
    return total + current


def cross_validate_amounts(numeric_str: str, words_str: str) -> dict:
    FA_TO_EN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    try:
        num_val = int(re.sub(r"[,،٬]", "", numeric_str).translate(FA_TO_EN))
    except ValueError:
        return {"match": None, "reason": "cannot parse numeric"}

    word_val = words_to_int(words_str)
    if word_val is None:
        return {"match": None, "reason": "cannot parse words"}

    match = num_val == word_val
    return {
        "match":        match,
        "numeric_int":  num_val,
        "words_int":    word_val,
        "discrepancy":  abs(num_val - word_val) if not match else 0,
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        ("amount_numeric", "۵۰۰،۰۰۰،۰۰۰ ریال abc"),
        ("amount_words",   "پانصد ملیون ریال"),      # typo: ملیون → میلیون
        ("account",        "۸۹۱-۲۸-x654197-4"),
        ("date",           "۱۴۰۳/۱۱/۱۵"),
        ("payee",          "سارا 123 نجفی"),
    ]
    for field, raw in samples:
        r = postprocess_field(field, raw)
        print(f"[{field}]")
        print(f"  raw:       {r.raw}")
        print(f"  corrected: {r.corrected}")
        print(f"  changed:   {r.changed}")
        print()

    print("Cross-validation demo:")
    cv = cross_validate_amounts(
        "500000000",
        "پانصد میلیون ریال"
    )
    print(cv)
