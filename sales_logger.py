"""FitMeal Sales Logger (S2).

Usage:
    python sales_logger.py --menu "ข้าวอกไก่ย่างซอสเทอริยากิ" --qty 2 --price 119

Reads GOOGLE_SHEETS_CREDENTIALS and TELEGRAM_BOT_TOKEN (or LINE_CHANNEL_TOKEN) from env.
Appends row [timestamp, menu, qty, price, total] to a Google Sheet,
then sends a notification via Telegram or LINE bot.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import gspread
import requests


DEFAULT_SHEET_NAME = "fitmeal-sheet"


def append_to_sheet(menu: str, qty: int, price: float) -> dict:
    """Append a sales row to the configured Google Sheet and return the saved row payload."""
    credentials_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS", "")
    sheet_name = os.environ.get("GOOGLE_SHEETS_SHEET_NAME", DEFAULT_SHEET_NAME)

    if not credentials_json:
        raise RuntimeError("GOOGLE_SHEETS_CREDENTIALS not set")

    try:
        credentials = json.loads(credentials_json)
        client = gspread.service_account_from_dict(credentials)
        spreadsheet = client.open(sheet_name)
        worksheet = spreadsheet.sheet1
    except Exception as exc:
        raise RuntimeError(
            f"Unable to open Google Sheet '{sheet_name}': {exc}") from exc

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = qty * price
    row = [timestamp, menu, qty, price, total]
    worksheet.append_row(row)
    return {"timestamp": timestamp, "menu": menu, "qty": qty, "price": price, "total": total}


def send_notification(message: str) -> str:
    """Send a notification through Telegram or LINE based on the available credentials."""
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    line_token = os.environ.get("LINE_CHANNEL_TOKEN", "")

    if telegram_token and telegram_chat_id:
        response = requests.post(
            f"https://api.telegram.org/bot{telegram_token}/sendMessage",
            data={"chat_id": telegram_chat_id, "text": message},
            timeout=10,
        )
        response.raise_for_status()
        return "telegram"

    if line_token:
        response = requests.post(
            "https://api.line.me/v2/bot/message/push",
            headers={
                "Authorization": f"Bearer {line_token}",
                "Content-Type": "application/json",
            },
            json={"to": os.environ.get("LINE_USER_ID", ""), "messages": [
                {"type": "text", "text": message}]},
            timeout=10,
        )
        response.raise_for_status()
        return "line"

    raise RuntimeError("No Telegram or LINE notification credentials found")


def main() -> int:
    parser = argparse.ArgumentParser(description="FitMeal Sales Logger")
    parser.add_argument("--menu", required=True, help="ชื่อเมนู")
    parser.add_argument("--qty", type=int, required=True, help="จำนวนกล่อง")
    parser.add_argument("--price", type=float, required=True, help="ราคาต่อกล่อง")
    args = parser.parse_args()

    try:
        row = append_to_sheet(args.menu, args.qty, args.price)
        total = row["total"]
    except Exception as exc:
        print(f"[ERROR] บันทึก Sheet ล้มเหลว: {exc}", file=sys.stderr)
        print("[HINT] ตรวจ GOOGLE_SHEETS_CREDENTIALS และ share Sheet กับ service account email", file=sys.stderr)
        return 1

    try:
        provider = send_notification(
            f"บันทึก {args.menu} x{args.qty} = {total} บาท")
    except Exception as exc:
        print(
            f"[WARN] บันทึก Sheet สำเร็จแต่ส่งแจ้งเตือนล้มเหลว: {exc}", file=sys.stderr)
        return 0

    print(f"[OK] บันทึกและแจ้งเตือนผ่าน {provider} เรียบร้อย ยอด {total} บาท")
    return 0


if __name__ == "__main__":
    sys.exit(main())
