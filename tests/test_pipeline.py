"""
Persian Banking OCR — Test Suite
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import json
import numpy as np


# ── NLP post-processor ─────────────────────────────────────────────────────

from nlp.postprocessor import (
    postprocess_field, cross_validate_amounts,
    clean_amount_numeric, clean_amount_words,
    clean_account, clean_date, normalise,
)

def test_normalise_arabic_digits():
    assert normalise("٣") == "۳"

def test_normalise_arabic_kaf():
    assert "ک" in normalise("كلمه")

def test_clean_amount_numeric_strips_latin():
    result = clean_amount_numeric("۵۰۰،۰۰۰ ریال abc")
    assert "a" not in result and "b" not in result

def test_clean_amount_words_corrects_typo():
    result = clean_amount_words("پانصد ملیون ریال")
    assert "میلیون" in result

def test_clean_account_removes_letters():
    result = clean_account("۸۹۱-۲۸-x654197-4")
    for ch in result:
        assert ch in "۰۱۲۳۴۵۶۷۸۹-"

def test_clean_date_keeps_digits_and_slash():
    result = clean_date("۱۴۰۳/۱۱/۱۵")
    for ch in result:
        assert ch in "۰۱۲۳۴۵۶۷۸۹/"

def test_cross_validate_match():
    cv = cross_validate_amounts("500000000", "پانصد میلیون ریال")
    assert cv["match"] is True

def test_cross_validate_mismatch():
    cv = cross_validate_amounts("500000000", "صد میلیون ریال")
    assert cv["match"] is False
    assert cv["discrepancy"] == 400_000_000

def test_postprocess_field_returns_result():
    r = postprocess_field("date", "۱۴۰۳/۱۱/۱۵")
    assert r.field == "date"
    assert r.corrected == "۱۴۰۳/۱۱/۱۵"


# ── Preprocessing ──────────────────────────────────────────────────────────

from preprocessing.pipeline import detect_skew, adaptive_threshold

def test_detect_skew_returns_float():
    gray = np.zeros((200, 400), dtype=np.uint8)
    angle = detect_skew(gray)
    assert isinstance(angle, float)

def test_adaptive_threshold_shape():
    gray = np.random.randint(0, 255, (200, 400), dtype=np.uint8)
    binary = adaptive_threshold(gray)
    assert binary.shape == gray.shape
    assert set(np.unique(binary)).issubset({0, 255})


# ── Layout detector ────────────────────────────────────────────────────────

from layout.detector import crop_fields, FIELD_TEMPLATES

def test_crop_fields_count():
    binary = np.full((420, 900), 255, dtype=np.uint8)
    crops = crop_fields(binary)
    assert len(crops) == len(FIELD_TEMPLATES)

def test_crop_fields_names():
    binary = np.full((420, 900), 255, dtype=np.uint8)
    crops  = crop_fields(binary)
    names  = {c.name for c in crops}
    assert names == set(FIELD_TEMPLATES.keys())


# ── Synthetic data manifest ────────────────────────────────────────────────

def test_manifest_exists():
    p = Path("data/synthetic/manifest.json")
    assert p.exists(), "Run scripts/generate_synthetic_data.py first"

def test_manifest_structure():
    p = Path("data/synthetic/manifest.json")
    if not p.exists():
        pytest.skip("Manifest not found")
    with open(p, encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest) >= 10
    sample = manifest[0]
    for key in ("id", "file", "degraded", "fields"):
        assert key in sample
    for field in ("date", "amount_numeric", "amount_words", "account", "payee"):
        assert field in sample["fields"]
