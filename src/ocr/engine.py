"""
Persian Banking OCR — Engine
- فیلدهای عددی (مبلغ، حساب، تاریخ): مدل ONNX آموزش‌دیده روی Hoda
- فیلدهای متنی (مبلغ حروفی، نام): EasyOCR با پشتیبانی فارسی
"""

import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

# ─── مسیر مدل ONNX ────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "persian_digit_ocr.onnx"
PERSIAN    = "۰۱۲۳۴۵۶۷۸۹"
IMG_SIZE   = 32

# فیلدهایی که فقط رقم دارن → مدل Hoda
NUMERIC_FIELDS = {"amount_numeric", "account", "date"}
# فیلدهایی که متن فارسی دارن → EasyOCR
TEXT_FIELDS    = {"amount_words", "payee"}

# ─── Session های lazy-load ────────────────────────────────────────────────
_onnx_session = None
_easy_reader  = None


def get_onnx_session():
    global _onnx_session
    if _onnx_session is None:
        print(f"⏳ لود مدل ONNX: {MODEL_PATH.name}")
        _onnx_session = ort.InferenceSession(str(MODEL_PATH))
        print("✅ مدل ONNX لود شد")
    return _onnx_session


def get_easy_reader():
    global _easy_reader
    if _easy_reader is None:
        import easyocr
        print("⏳ لود EasyOCR (فارسی + انگلیسی)...")
        _easy_reader = easyocr.Reader(["fa", "en"], gpu=False)
        print("✅ EasyOCR لود شد")
    return _easy_reader


# ─── تشخیص یه رقم با ONNX ────────────────────────────────────────────────

def predict_digit(img: np.ndarray) -> tuple[str, float]:
    """یه تصویر رقم می‌گیره و رقم فارسی + confidence برمی‌گردونه"""
    session = get_onnx_session()
    img = cv2.resize(img.astype(np.float32), (IMG_SIZE, IMG_SIZE))
    if img.max() > 1:
        img = img / 255.0
    tensor = img[None, None, :, :].astype(np.float32)
    logits = session.run(None, {"image": tensor})[0][0]
    probs  = np.exp(logits) / np.exp(logits).sum()
    cls    = probs.argmax()
    return PERSIAN[cls], float(probs[cls] * 100)


# ─── OCR عددی با Hoda ────────────────────────────────────────────────────

def ocr_numeric(crop_img: np.ndarray) -> tuple[str, float]:
    """خواندن فیلدهای عددی رقم به رقم با مدل Hoda"""
    if crop_img is None or crop_img.size == 0:
        return "", 50.0

    # تبدیل به uint8
    if crop_img.dtype != np.uint8:
        mn, mx = crop_img.min(), crop_img.max()
        crop_img = ((crop_img - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)

    # binarize
    _, binary = cv2.threshold(
        crop_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # پیدا کردن contour هر رقم
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return "", 55.0

    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [(x, y, w, h) for x, y, w, h in boxes if w > 3 and h > 5]
    if not boxes:
        return "", 55.0

    # راست به چپ (فارسی)
    boxes.sort(key=lambda b: b[0], reverse=True)

    chars, confs = [], []
    for x, y, w, h in boxes:
        roi = crop_img[max(0, y-2):y+h+2, max(0, x-2):x+w+2]
        if roi.size == 0:
            continue
        persian, conf = predict_digit(roi)
        chars.append(persian)
        confs.append(conf)

    if not chars:
        return "", 55.0

    return "".join(chars), float(np.mean(confs))


# ─── OCR متنی با EasyOCR ─────────────────────────────────────────────────

def ocr_text(crop_img: np.ndarray) -> tuple[str, float]:
    """خواندن فیلدهای متنی فارسی با EasyOCR"""
    if crop_img is None or crop_img.size == 0:
        return "", 50.0

    reader  = get_easy_reader()
    results = reader.readtext(crop_img, detail=1, paragraph=False)

    if not results:
        return "", 50.0

    texts, confs = [], []
    for (_, text, conf) in results:
        if text.strip():
            texts.append(text.strip())
            confs.append(conf * 100)

    if not texts:
        return "", 50.0

    # فارسی راست به چپ — ترتیب معکوس
    full_text = " ".join(reversed(texts))
    avg_conf  = float(np.mean(confs))
    return full_text, avg_conf


# ─── تابع اصلی ───────────────────────────────────────────────────────────

def ocr_field(field_name: str, crop_img: np.ndarray) -> tuple[str, float]:
    """
    بر اساس نوع فیلد، مدل مناسب رو انتخاب می‌کنه:
    - عددی (مبلغ، حساب، تاریخ) → Hoda ONNX
    - متنی (مبلغ حروفی، نام)    → EasyOCR
    """
    if field_name in NUMERIC_FIELDS:
        return ocr_numeric(crop_img)
    elif field_name in TEXT_FIELDS:
        return ocr_text(crop_img)
    else:
        return ocr_numeric(crop_img)
