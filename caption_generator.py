"""FitMeal Caption Generator (S1).

Usage:
    python caption_generator.py

Reads GOOGLE_API_KEY from env. Generates a Thai caption for a FitMeal menu item.
"""

import os
import sys

from dotenv import load_dotenv
from google import genai


PROMPT_TEMPLATE = """\
คุณคือ social media manager ของร้าน FitMeal ร้านอาหารสุขภาพ คาร์บต่ำ โปรตีนสูง คุมแคลอรี

จงเขียนแคปชั่นภาษาไทย 2 ถึง 3 ประโยคโปรโมตเมนู: {menu}

เงื่อนไข:
- โทนสดใส เหมาะกับคนรักสุขภาพ ใช้คำง่าย ใส่ emoji ได้
- เน้นจุดเด่น เช่น แคลอรีต่ำ โปรตีนสูง หรือคาร์บต่ำ
- ต้องมี call-to-action ปิดท้าย เช่น สั่งเลย หรือ ทักแชท LINE OA (@FitMealThailand)
- ห้ามใช้ em dash
"""

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
MODEL_FALLBACKS = [
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-flash-latest",
]


def generate_caption(menu: str, api_key: str | None = None) -> str:
    """Generate a Thai caption for the given FitMeal menu item."""
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in env or argument")

    load_dotenv()
    preferred_model = os.environ.get(
        "GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    model_candidates = [preferred_model] if preferred_model else []
    for model_name in MODEL_FALLBACKS:
        if model_name not in model_candidates:
            model_candidates.append(model_name)

    client = genai.Client(api_key=key)
    last_error: Exception | None = None
    for model_name in model_candidates:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=PROMPT_TEMPLATE.format(menu=menu),
            )
            if response.text:
                return response.text.strip()
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"All Gemini model candidates failed. Last error: {last_error}")


def main() -> int:
    load_dotenv()
    menu = input("เมนูสุขภาพที่จะโปรโมต: ").strip()
    if not menu:
        print("กรุณาใส่ชื่อเมนู")
        return 1
    caption = generate_caption(menu)
    print()
    print(caption)
    return 0


if __name__ == "__main__":
    sys.exit(main())
