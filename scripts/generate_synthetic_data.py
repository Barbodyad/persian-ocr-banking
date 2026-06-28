"""
Persian Banking OCR — Synthetic Data Generator
Generates realistic Persian bank cheque/form images for training and demo.
"""

import os
import random
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Persian number / word helpers
# ---------------------------------------------------------------------------

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"

def to_persian_digits(n: int) -> str:
    return "".join(PERSIAN_DIGITS[int(d)] for d in str(n))

UNITS = ["", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه",
         "ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده", "شانزده",
         "هفده", "هجده", "نوزده"]
TENS  = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
HUNDREDS = ["", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد",
            "هفتصد", "هشتصد", "نهصد"]

def _group(n: int) -> str:
    if n == 0:
        return ""
    if n < 20:
        return UNITS[n]
    if n < 100:
        t, u = divmod(n, 10)
        return TENS[t] + (" و " + UNITS[u] if u else "")
    h, rest = divmod(n, 100)
    return HUNDREDS[h] + (" و " + _group(rest) if rest else "")

def amount_to_words(amount: int) -> str:
    if amount == 0:
        return "صفر"
    parts = []
    scales = [
        (1_000_000_000_000, "تریلیون"),
        (1_000_000_000,     "میلیارد"),
        (1_000_000,         "میلیون"),
        (1_000,             "هزار"),
        (1,                 ""),
    ]
    for val, label in scales:
        if amount >= val:
            q, amount = divmod(amount, val)
            chunk = _group(q)
            parts.append((chunk + " " + label).strip() if label else chunk)
    return " و ".join(p for p in parts if p) + " ریال"

# ---------------------------------------------------------------------------
# Random banking data
# ---------------------------------------------------------------------------

BANK_NAMES = ["بانک ملی", "بانک صادرات", "بانک پارسیان", "بانک ملت", "بانک آینده"]
BRANCH_NAMES = ["شعبه مرکزی", "شعبه ولیعصر", "شعبه انقلاب", "شعبه تجریش"]

def random_account() -> str:
    parts = [str(random.randint(100, 999)),
             str(random.randint(10, 99)),
             str(random.randint(100000, 999999)),
             str(random.randint(1, 9))]
    return "-".join(parts)

def random_amount() -> int:
    choices = [
        random.randint(1_000_000, 50_000_000),
        random.randint(50_000_000, 500_000_000),
        random.randint(500_000_000, 5_000_000_000),
    ]
    return random.choice(choices)

def random_date() -> str:
    y = random.randint(1400, 1403)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{to_persian_digits(y)}/{to_persian_digits(m):>2}/{to_persian_digits(d):>2}".replace(" ", "۰")

def random_name() -> str:
    firsts = ["علی", "محمد", "فاطمه", "زینب", "حسین", "رضا", "مریم", "سارا"]
    lasts  = ["احمدی", "محمدی", "رضایی", "حسینی", "کریمی", "نجفی", "موسوی"]
    return random.choice(firsts) + " " + random.choice(lasts)

# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------

def reshape(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))

def draw_rtl(draw: ImageDraw.ImageDraw, xy, text: str, font, fill=(20, 20, 20)):
    draw.text(xy, reshape(text), font=font, fill=fill)

# ---------------------------------------------------------------------------
# Image synthesiser
# ---------------------------------------------------------------------------

def add_noise(img: Image.Image, level: float = 0.03) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, level * 255, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def add_skew(img: Image.Image, max_deg: float = 2.5) -> Image.Image:
    angle = random.uniform(-max_deg, max_deg)
    return img.rotate(angle, fillcolor=(245, 240, 230), expand=False)

def add_shadow(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    # Gradient shadow from one corner
    sx, sy = random.uniform(0.3, 0.8), random.uniform(0.3, 0.8)
    X = np.linspace(0, 1, w)
    Y = np.linspace(0, 1, h)
    gx = 1 - sx * X
    gy = 1 - sy * Y[:, None]
    mask = (gx * gy).clip(0.7, 1.0)
    arr = (arr * mask[..., None]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def make_cheque(sample_id: int, degraded: bool = True) -> dict:
    W, H = 900, 420
    bg = (245, 240, 230)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    # Border
    draw.rectangle([8, 8, W - 9, H - 9], outline=(160, 130, 90), width=2)

    # Header bar
    draw.rectangle([8, 8, W - 9, 60], fill=(200, 170, 110))

    # Try to use a system font; fall back to default
    try:
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font_lg = font_md = font_sm = ImageFont.load_default()

    # Bank / branch info
    bank   = random.choice(BANK_NAMES)
    branch = random.choice(BRANCH_NAMES)
    draw_rtl(draw, (W - 30, 18), bank,   font_lg, fill=(60, 30, 10))
    draw_rtl(draw, (W - 30, 40), branch, font_sm, fill=(80, 50, 20))

    # Field lines
    line_color = (160, 140, 100)
    for y in [100, 160, 220, 280, 340]:
        draw.line([(40, y), (W - 40, y)], fill=line_color, width=1)

    # Labels
    labels = {
        80:  "تاریخ :",
        140: "مبلغ به عدد :",
        200: "مبلغ به حروف :",
        260: "شماره حساب :",
        320: "در وجه :",
    }
    for y, label in labels.items():
        draw_rtl(draw, (W - 50, y), label, font_sm, fill=(100, 80, 50))

    # Values
    amount  = random_amount()
    account = random_account()
    date    = random_date()
    payee   = random_name()
    words   = amount_to_words(amount)
    amount_str = to_persian_digits(amount)
    # Format with comma-like separators using Persian digits
    formatted_amount = ""
    s = str(amount)[::-1]
    groups = [s[i:i+3] for i in range(0, len(s), 3)]
    formatted_amount = to_persian_digits(int(",".join(groups[::-1]).replace(",", "")))

    values = {
        80:  date,
        140: to_persian_digits(amount),
        200: words,
        260: account,
        320: payee,
    }
    for y, val in values.items():
        draw_rtl(draw, (W - 200, y), val, font_md, fill=(15, 15, 15))

    # Signature box
    draw.rectangle([50, 350, 250, 400], outline=line_color, width=1)
    draw_rtl(draw, (240, 355), "امضاء :", font_sm, fill=(100, 80, 50))

    # Cheque number (bottom)
    cheque_no = to_persian_digits(random.randint(100000, 999999))
    draw_rtl(draw, (W // 2, H - 30), f"شماره چک : {cheque_no}", font_sm, fill=(80, 60, 40))

    # Degradation
    if degraded:
        img = add_noise(img, level=random.uniform(0.01, 0.04))
        img = add_shadow(img)
        img = add_skew(img, max_deg=random.uniform(0.5, 3.0))

    fname = f"cheque_{sample_id:04d}{'_degraded' if degraded else '_clean'}.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    img.save(fpath, dpi=(300, 300))

    return {
        "id":       sample_id,
        "file":     fname,
        "degraded": degraded,
        "fields": {
            "date":           date,
            "amount_numeric": to_persian_digits(amount),
            "amount_words":   words,
            "account":        account,
            "payee":          payee,
            "cheque_no":      cheque_no,
            "bank":           bank,
        }
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N = 20  # samples per type
    records = []
    for i in range(N):
        records.append(make_cheque(i,       degraded=False))
        records.append(make_cheque(i + N,   degraded=True))

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(records)} samples → {OUTPUT_DIR}")
    print(f"Manifest  → {manifest_path}")
