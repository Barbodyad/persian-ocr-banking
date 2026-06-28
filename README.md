# 🏦 Persian Banking OCR System

> End-to-end pipeline for reading Persian handwritten bank cheques and forms — from raw scanned images to structured JSON — with Human-in-the-Loop fallback.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-red.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📐 System Architecture

```
┌──────────────────────────────────────────────────────┐
│  Phase 1 · Image Preprocessing                       │
│  Deskew (Hough) → Shadow Removal → Adaptive Binarise │
└───────────────────────┬──────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────┐
│  Phase 2 · Layout Analysis                           │
│  YOLOv8 / Template Crop → Field Patches              │
└───────────────────────┬──────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────┐
│  Phase 3 · Core OCR Engine                           │
│  TrOCR / CRNN+LSTM → Raw Text + Confidence Score     │
└───────────────────────┬──────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────┐
│  Phase 4 · NLP Post-Processing                       │
│  Levenshtein Correction → Regex Filters              │
│  Cross-validation: amount_numeric ↔ amount_words     │
└───────────────────────┬──────────────────────────────┘
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
| **Adaptive binarisation** | Gaussian adaptive threshold — handles uneven light |
| **Layout detection** | Template-based crop + contour fallback |
| **NLP correction** | Levenshtein distance against Persian banking vocabulary |
| **Cross-validation** | `amount_numeric` ↔ `amount_words` integer comparison |
| **Confidence routing** | Auto-route ≥ 85% to core banking; < 85% to human review |
| **Synthetic data** | Generator for 40+ realistic Persian cheque images |
| **REST API** | FastAPI with `/ocr/upload`, `/ocr/demo`, `/health` |

---

## 🗂️ Project Structure

```
persian-ocr-banking/
├── data/
│   ├── raw/                  # Original scanned images
│   ├── processed/            # Field crops after layout analysis
│   └── synthetic/            # Auto-generated training samples
│       └── manifest.json     # Ground-truth labels
├── src/
│   ├── preprocessing/
│   │   └── pipeline.py       # Phase 1: deskew, shadow, binarise
│   ├── layout/
│   │   └── detector.py       # Phase 2: field crop & contour detection
│   ├── ocr/                  # Phase 3: TrOCR / CRNN+LSTM (plug-in)
│   ├── nlp/
│   │   └── postprocessor.py  # Phase 4: Levenshtein + regex cleaners
│   └── api/
│       └── main.py           # FastAPI application
├── scripts/
│   └── generate_synthetic_data.py
├── tests/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/persian-ocr-banking.git
cd persian-ocr-banking
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

### 4. Test the demo endpoint

```bash
# Random degraded sample — full pipeline
curl -X POST http://localhost:8000/ocr/demo \
     -H "Content-Type: application/json" \
     -d '{}' | python -m json.tool

# Upload your own cheque image
curl -X POST http://localhost:8000/ocr/upload \
     -F "file=@my_cheque.jpg"
```

### 5. Interactive docs

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the Swagger UI.

---

## 📊 Sample API Response

```json
{
  "status": "success",
  "processing_time_sec": 0.046,
  "pipeline": {
    "skew_angle": 1.0,
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

## 🔬 Data Pipeline

### Synthetic Data Generator

The generator (`scripts/generate_synthetic_data.py`) produces realistic Persian cheques with:

- Random amounts (1M – 5B Rial) with correct Persian word form
- Random dates in Jalali calendar (Persian digits)
- Random account numbers, bank names, payee names
- Clean + degraded variants (noise, shadow gradient, rotation)

### Ground-Truth Manifest

Each sample in `data/synthetic/manifest.json` contains full field labels for evaluation and fine-tuning.

---

## 🧠 NLP Post-Processing

### Levenshtein Correction

Each OCR token is matched against a 50-word Persian banking vocabulary. Tokens within edit distance 2 are auto-corrected:

```
ملیون   →  میلیون   (edit distance 1)
هزارr   →  هزار     (edit distance 1)
```

### Field-Specific Rules

| Field | Rule |
|---|---|
| `amount_numeric` | Keep only `[۰-۹,،٬]` |
| `amount_words` | Correct each token vs. vocabulary |
| `account` | Keep only digits and `-`, convert Arabic → Persian |
| `date` | Keep only `[۰-۹/]` |
| `payee` | Keep only `[\u0600-\u06FF\s]` |

### Cross-Validation

The numeric amount and word amount are independently parsed to integers and compared. A mismatch triggers human review regardless of confidence score.

---

## 🔌 Plugging in a Real OCR Model

Replace the `mock_ocr()` function in `src/api/main.py` with your model call:

```python
# Example: TrOCR (Microsoft)
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model     = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def real_ocr(field_image: np.ndarray) -> tuple[str, float]:
    pil_img    = Image.fromarray(field_image)
    pixel_vals = processor(pil_img, return_tensors="pt").pixel_values
    ids        = model.generate(pixel_vals)
    text       = processor.batch_decode(ids, skip_special_tokens=True)[0]
    # Confidence from beam scores or logits
    return text, 90.0
```

---

## 🛡️ Security & Compliance

- All image files are processed in-memory and deleted immediately after OCR
- No PII is logged
- Designed for air-gapped deployment within bank network
- Audit log hooks ready in `src/api/main.py`

---

## 📄 License

MIT © 2024 — Persian Banking OCR Project
