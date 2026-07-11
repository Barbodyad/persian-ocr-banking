# 🏦 Persian Banking OCR System

> An end-to-end AI pipeline for reading Persian handwritten bank cheques — from raw scanned images to structured JSON output — with a Human-in-the-Loop fallback for low-confidence results.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-red.svg)](https://opencv.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![CI](https://github.com/Barbodyad/persian-ocr-banking/actions/workflows/ci.yml/badge.svg)](https://github.com/Barbodyad/persian-ocr-banking/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📸 Demo

<img width="385" height="189" alt="image" src="https://github.com/user-attachments/assets/02fcca13-01c2-4cb4-9a8a-2db8f0e1be1f" />


*Real cheque image processed by the pipeline — fields detected and highlighted automatically*

---

## 🎯 What It Does

This system takes a photo or scan of a Persian bank cheque and automatically extracts:

| Field | Example |
|---|---|
| 📅 Date | `۱۳۹۲/۰۹/۰۹` |
| 💰 Amount (numeric) | `۱۲۰,۰۰۰,۰۰۰` |
| ✍️ Amount (words) | `یکصد و بیست میلیون ریال` |
| 🏦 Account number | `123-456-789` |
| 👤 Payee | `آقای محمد` |

High-confidence results (≥ 85%) are sent directly to core banking as JSON. Low-confidence results are routed to a human reviewer.

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
│  Contour Detection → Field Patches                  │
│  (date / amount_numeric / amount_words /            │
│   account / payee)                                  │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Phase 3 · Dual OCR Engine                          │
│  Numeric fields → CNN trained on Hoda Dataset (98%) │
│  Text fields    → EasyOCR + Keyword Field Mapping   │
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
| **Dual OCR engine** | Hoda CNN (98% accuracy) for digits · EasyOCR for Persian text |
| **Keyword field mapping** | Detects fields by Persian keywords, not fixed coordinates |
| **NLP correction** | Levenshtein distance against 50-word Persian banking vocabulary |
| **Cross-validation** | `amount_numeric` ↔ `amount_words` integer comparison |
| **Confidence routing** | ≥ 85% → core banking · < 85% → human review |
| **Synthetic data generator** | 40+ realistic Persian cheque images with ground-truth labels |
| **REST API** | FastAPI with `/ocr/upload`, `/ocr/vision`, `/ocr/demo` |
| **ONNX export** | Lightweight model deployment without GPU |

---

## 🧠 OCR Engine — Hoda Dataset

The digit recognition model was trained on the **Hoda Dataset** — the standard Persian handwritten digit benchmark:

- **102,352** digit samples from **12,000** real Iranian forms
- **CNN architecture**: 3 convolutional blocks + BatchNorm + Dropout
- **Test accuracy: 98%+**
- **Exported to ONNX** for fast CPU inference
- Training notebook: `Persian_Banking_OCR_Hoda.ipynb` (Google Colab, T4 GPU)

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
│   │   └── detector.py       # Phase 2: contour + field detection
│   ├── ocr/
│   │   └── engine.py         # Phase 3: Hoda CNN + EasyOCR dispatcher
│   ├── nlp/
│   │   └── postprocessor.py  # Phase 4: Levenshtein + regex cleaners
│   └── api/
│       └── main.py           # FastAPI application
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── smart_ocr.py          # Smart OCR with keyword field mapping
│   └── demo_visualization.py # Visual OCR result display
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
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python scripts/generate_synthetic_data.py
```

### 3. Run the API

```bash
cd src
uvicorn api.main:app --reload --port 8000
```

### 4. Test with Swagger UI

```
http://localhost:8000/docs
```

### 5. Run smart OCR on a real cheque

```bash
python scripts/smart_ocr.py path/to/cheque.jpg
```

---

## 📊 Sample API Response

```json
{
  "status": "success",
  "processing_time_sec": 0.046,
  "fields": {
    "date":           { "corrected": "۱۳۹۲/۰۹/۰۹",        "confidence": 92.1 },
    "amount_numeric": { "corrected": "۱۲۰۰۰۰۰۰۰",          "confidence": 94.3 },
    "amount_words":   { "corrected": "یکصد و بیست میلیون ریال", "confidence": 91.7 },
    "account":        { "corrected": "۱۲۳۴۵۶۷۸۹",           "confidence": 95.0 },
    "payee":          { "corrected": "آقای محمد",             "confidence": 88.5 }
  },
  "cross_validation": { "match": true },
  "avg_confidence": 92.3,
  "route": "json_to_core_banking"
}
```

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
# 15 passed ✅
```

---

## 🔌 Plugging in a Better OCR Model

The `src/ocr/engine.py` is modular — swap in any model:

```python
# Fine-tuned TrOCR for Persian banking
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def ocr_field(field_name, crop_img):
    # your fine-tuned model here
    return text, confidence
```

> **Note:** For production-level accuracy on real Persian bank cheques, fine-tuning TrOCR on 500+ labeled cheque images is recommended (estimated accuracy: 95%+).

---

## 🛡️ Security & Compliance

- All uploaded images processed in-memory and deleted immediately
- No PII logged
- Designed for air-gapped bank network deployment
- Confidence threshold configurable via `CONFIDENCE_THRESHOLD`

---

## 📄 License

MIT © 2024 — Persian Banking OCR Project
