# 🏦 Persian Banking OCR System

> An end-to-end AI pipeline for reading Persian handwritten bank cheques — from raw scanned images to structured JSON output — with a Human-in-the-Loop fallback for low-confidence results.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-red.svg)](https://opencv.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────┐
│  Phase 1 · Image Preprocessing                      │
│  Deskew (Hough) → Shadow Removal → Binarisation     │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Phase 2 · Layout Analysis                          │
│  Template Crop → Field Patches                      │
│  (date / amount_numeric / amount_words /            │
│   account / payee)                                  │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Phase 3 · Dual OCR Engine                          │
│  Numeric fields → CNN trained on Hoda Dataset       │
│  Text fields    → EasyOCR (Persian + English)       │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Phase 4 · NLP Post-Processing                      │
│  Levenshtein Correction → Regex Filters             │
│  Cross-validation: amount_numeric ↔ amount_words    │
└──────────────────────┬──────────────────────────────┘
                       ▼
             ┌─────────┴──────────┐
       conf ≥ 85%           conf < 85%
             │                   │
             ▼                   ▼
   JSON → Core Banking   Human-in-the-Loop Panel
```

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| **Deskew** | Hough transform — corrects up to ±5° rotation |
| **Shadow removal** | Morphological background normalisation |
| **Adaptive binarisation** | Gaussian adaptive threshold for uneven lighting |
| **Dual OCR engine** | Hoda CNN for digits · EasyOCR for Persian text |
| **NLP correction** | Levenshtein distance against Persian banking vocabulary |
| **Cross-validation** | `amount_numeric` ↔ `amount_words` integer comparison |
| **Confidence routing** | ≥ 85% → core banking · < 85% → human review |
| **Synthetic data generator** | 40+ realistic Persian cheque images with ground-truth labels |
| **REST API** | FastAPI with `/ocr/upload`, `/ocr/demo`, `/health` |
| **ONNX export** | Lightweight model deployment without PyTorch |

---

## 🧠 OCR Engine — Hoda Dataset

The digit recognition model was trained on the **Hoda Dataset** — the standard Persian handwritten digit benchmark:

- **102,352** digit samples extracted from **12,000** real Iranian forms
- **CNN architecture**: 3 convolutional blocks + BatchNorm + Dropout
- **Test accuracy: 98%+** on the Hoda test split
- **Exported to ONNX** for fast CPU inference (no GPU required)
- **Training notebook**: `Persian_Banking_OCR_Hoda.ipynb` (Google Colab, T4 GPU)

---

## 🗂️ Project Structure

```
persian-ocr-banking/
├── data/
│   ├── raw/                  # Original scanned images
│   ├── processed/            # Field crops after layout analysis
│   └── synthetic/            # Auto-generated training samples
│       └── manifest.json     # Ground-truth labels
├── models/
│   ├── persian_digit_ocr.onnx       # Trained ONNX model
│   └── persian_digit_ocr.onnx.data  # Model weights
├── src/
│   ├── preprocessing/
│   │   └── pipeline.py       # Phase 1: deskew, shadow, binarise
│   ├── layout/
│   │   └── detector.py       # Phase 2: field crop & contour detection
│   ├── ocr/
│   │   └── engine.py         # Phase 3: Hoda CNN + EasyOCR dispatcher
│   ├── nlp/
│   │   └── postprocessor.py  # Phase 4: Levenshtein + regex cleaners
│   └── api/
│       └── main.py           # FastAPI application
├── scripts/
│   └── generate_synthetic_data.py
├── tests/
│   └── test_pipeline.py      # 15 automated tests
├── .github/
│   └── workflows/ci.yml      # GitHub Actions CI
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Barbodyad/persian-ocr-banking.git
cd persian-ocr-banking
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python scripts/generate_synthetic_data.py
# → 40 cheque images in data/synthetic/
# → manifest.json with full ground-truth labels
```

### 3. Run the API

```bash
cd src
uvicorn api.main:app --reload --port 8000
```

### 4. Test

```bash
# Swagger UI
http://localhost:8000/docs

# Random synthetic sample
http://localhost:8000/ocr/demo/random

# Upload a real cheque image
curl -X POST http://localhost:8000/ocr/upload -F "file=@cheque.jpg"
```

---

## 📊 Sample API Response

```json
{
  "status": "success",
  "processing_time_sec": 0.046,
  "pipeline": {
    "skew_angle": 1.2,
    "had_shadow": true,
    "fields_found": 5
  },
  "fields": {
    "date":           { "corrected": "۱۴۰۳/۱۱/۱۶", "confidence": 92.1 },
    "amount_numeric": { "corrected": "۵۰۰۰۰۰۰۰۰",  "confidence": 94.3 },
    "amount_words":   { "corrected": "پانصد میلیون ریال", "confidence": 91.7 },
    "account":        { "corrected": "۸۹۱-۲۸-۶۵۴۱۹۷-۴", "confidence": 95.0 },
    "payee":          { "corrected": "سارا نجفی",    "confidence": 88.5 }
  },
  "cross_validation": { "match": true, "numeric_int": 500000000, "words_int": 500000000 },
  "avg_confidence": 92.3,
  "route": "json_to_core_banking"
}
```

---

## 🔬 NLP Post-Processing

### Levenshtein Correction

Each OCR token is matched against a 50-word Persian banking vocabulary. Tokens within edit distance 2 are auto-corrected:

```
ملیون  →  میلیون   (edit distance 1)
هزارr  →  هزار     (edit distance 1)
```

### Field-Specific Rules

| Field | Rule |
|---|---|
| `amount_numeric` | Keep only `[۰-۹,،٬]` |
| `amount_words` | Correct each token vs. vocabulary |
| `account` | Digits and `-` only, convert Arabic → Persian |
| `date` | Keep only `[۰-۹/]` |
| `payee` | Keep only Persian Unicode range |

### Cross-Validation

The numeric amount and word amount are independently parsed to integers and compared. A mismatch triggers human review regardless of confidence score.

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
# → 15 passed
```

---

## 🔌 Replacing the OCR Engine

The `src/ocr/engine.py` is modular — swap in any model:

```python
# Example: fine-tuned TrOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("your-finetuned-model")
model     = VisionEncoderDecoderModel.from_pretrained("your-finetuned-model")

def ocr_field(field_name, crop_img):
    pil_img    = Image.fromarray(crop_img)
    pixel_vals = processor(pil_img, return_tensors="pt").pixel_values
    ids        = model.generate(pixel_vals)
    text       = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text, 90.0
```

---

## 🛡️ Security & Compliance

- All uploaded images are processed in-memory and deleted immediately
- No PII is logged
- Designed for air-gapped deployment within bank network
- Audit log hooks available in `src/api/main.py`
- Confidence threshold configurable via `CONFIDENCE_THRESHOLD`

---

## 📄 License

MIT © 2024 — Persian Banking OCR Project
