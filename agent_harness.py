"""MilkLab Agent Harness (S2).

Usage:
    python agent_harness.py --cmd "บันทึกขายนมหมี 2 ขวด ขวดละ 65"

รับคำสั่งภาษาไทย ส่งให้ Gemini พร้อม tool schema parse response เป็น tool call
เรียก tool จริง print trace log
"""

import argparse
import json
import os
import re
import sys

from dotenv import load_dotenv
from google import genai

from sales_logger import append_to_sheet, send_notification
import gspread


TOOL_SCHEMA = [
    {
        "name": "log_sale",
        "description": "บันทึกการขายลง Google Sheets และส่ง notification",
        "parameters": {
            "type": "object",
            "properties": {
                "menu": {"type": "string", "description": "ชื่อเมนู"},
                "qty": {"type": "integer", "description": "จำนวนที่ขาย"},
                "price": {"type": "number", "description": "ราคาต่อหน่วย"},
            },
            "required": ["menu", "qty", "price"],
        },
    },
    {
        "name": "query_sales",
        "description": "ดูยอดขายของวันที่ระบุ",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "วันที่ format YYYY-MM-DD"},
            },
            "required": ["date"],
        },
    },
    {
        "name": "send_alert",
        "description": "ส่ง message แจ้งเตือนผ่าน Bot",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
    },
]

DEFAULT_SHEET_NAME = os.environ.get(
    "GOOGLE_SHEETS_SHEET_NAME", "milklab-sheet")
DEFAULT_MODEL = "gemini-2.5-flash"
MODEL_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
]


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")

    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    raise ValueError("Unable to extract complete JSON object from response")


def _get_gemini_client(api_key: str | None = None):
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in env or argument")

    return genai.Client(api_key=key)


def _gemini_generate_json(cmd: str, api_key: str | None = None) -> dict:
    client = _get_gemini_client(api_key)
    preferred_model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL).strip()
    model_candidates = [preferred_model] if preferred_model else []
    for model in MODEL_FALLBACKS:
        if model not in model_candidates:
            model_candidates.append(model)

    prompt = (
        "You are a tool parser.\n"
        "Receive a Thai user command and return exactly one JSON object with keys 'tool' and 'args'.\n"
        "Use only the tools defined in TOOL_SCHEMA.\n"
        "Do not add any prose or explanation outside the JSON object.\n"
        f"TOOL_SCHEMA: {json.dumps(TOOL_SCHEMA, ensure_ascii=False)}\n"
        f"User command: {cmd}\n"
        "Output example: {\"tool\": \"log_sale\", \"args\": {\"menu\": \"นมหมีฮอกไกโด\", \"qty\": 2, \"price\": 65}}\n"
    )

    last_error: Exception | None = None
    for model_name in model_candidates:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    maxOutputTokens=256,
                    temperature=0.0,
                ),
            )
            if not response.text:
                continue

            raw_text = response.text.strip()
            json_text = _extract_json_object(raw_text)
            return json.loads(json_text)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Gemini parse_command failed. Last error: {last_error}"
    )


def _heuristic_parse_command(cmd: str) -> dict:
    normalized = cmd.strip().lower()

    if "ดูยอดขาย" in normalized or "ยอดขาย" in normalized or "วันที่" in normalized:
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", cmd)
        return {
            "tool": "query_sales",
            "args": {"date": date_match.group(1) if date_match else ""},
        }

    if "แจ้งเตือน" in normalized or "alert" in normalized:
        message = re.sub(
            r"^(?:ส่ง\s*)?(?:แจ้งเตือน|alert)(?:ว่า)?\s*", "", cmd).strip()
        return {
            "tool": "send_alert",
            "args": {"message": message or cmd.strip()},
        }

    qty_match = re.search(r"(\d+)\s*ขวด", cmd)
    price_match = re.search(r"ขวดละ\s*(\d+(?:\.\d+)?)", cmd)
    menu_match = re.search(
        r"(?:บันทึก\s*ขาย|บันทึก|ขาย)\s*(.+?)(?=\s+\d+\s*ขวด|\s+ขวดละ|\s+ราคา)", cmd)

    if qty_match and price_match:
        return {
            "tool": "log_sale",
            "args": {
                "menu": (menu_match.group(1).strip() if menu_match else cmd.strip()),
                "qty": int(qty_match.group(1)),
                "price": float(price_match.group(1)),
            },
        }

    raise RuntimeError("Unable to parse Thai command")


def parse_command(cmd: str, api_key: str | None = None) -> dict:
    """Parse a Thai command and return a tool call dictionary."""
    load_dotenv()
    try:
        tool_call = _gemini_generate_json(cmd, api_key=api_key)
    except Exception as exc:
        tool_call = _heuristic_parse_command(cmd)

    if not isinstance(tool_call, dict) or "tool" not in tool_call or "args" not in tool_call:
        raise RuntimeError("Gemini response does not contain tool and args")
    return tool_call


def _open_sales_sheet():
    credentials_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS", "")
    sheet_name = os.environ.get("GOOGLE_SHEETS_SHEET_NAME", DEFAULT_SHEET_NAME)
    if not credentials_json:
        raise RuntimeError("GOOGLE_SHEETS_CREDENTIALS not set")

    credentials = json.loads(credentials_json)
    client = gspread.service_account_from_dict(credentials)
    spreadsheet = client.open(sheet_name)
    return spreadsheet.sheet1


def _query_sales_by_date(date: str) -> float:
    sheet = _open_sales_sheet()
    rows = sheet.get_all_values()
    total = 0.0
    for row in rows:
        if not row or len(row) < 5:
            continue
        if row[0].startswith(date):
            try:
                total += float(row[4])
            except ValueError:
                continue
    return total


def dispatch_tool(tool_call: dict) -> str:
    tool_name = tool_call.get("tool")
    args = tool_call.get("args", {})

    if tool_name == "log_sale":
        menu = args.get("menu")
        qty = int(args.get("qty", 0))
        price = float(args.get("price", 0))
        row = append_to_sheet(menu, qty, price)
        try:
            provider = send_notification(
                f"บันทึก {menu} x{qty} = {row['total']} บาท")
            return f"บันทึกสำเร็จ ยอด {row['total']} บาท แจ้งเตือนผ่าน {provider}"
        except Exception as exc:
            return f"บันทึกสำเร็จ ยอด {row['total']} บาท แต่ส่งแจ้งเตือนล้ม: {exc}"

    if tool_name == "query_sales":
        date = args.get("date")
        if not date:
            raise RuntimeError("query_sales requires date")
        total = _query_sales_by_date(date)
        return f"ยอดขายวันที่ {date} = {total:.0f} บาท"

    if tool_name == "send_alert":
        message = args.get("message")
        if not message:
            raise RuntimeError("send_alert requires message")
        provider = send_notification(message)
        return f"แจ้งเตือนเรียบร้อยผ่าน {provider}"

    raise RuntimeError(f"Unknown tool: {tool_name}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True, help="คำสั่งภาษาไทย")
    args = parser.parse_args()

    print(f"[USER] {args.cmd}")
    try:
        tool_call = parse_command(args.cmd)
        print(f"[LLM]  tool={tool_call['tool']} args={tool_call['args']}")

        result = dispatch_tool(tool_call)
        print(f"[TOOL] {tool_call['tool']} {result}")
        print(f"[USER] ← {result}")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
