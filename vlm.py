import os
import base64
import json
import re
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from io import BytesIO

# =========================
# VLM 設定
# =========================
VLLM_LLM_MODEL2 = "gemma-4-26B-A4B-it"
VLLM_LLM_API_BASE2 = "http://10.2.5.111:8015/gemma-4-26B-A4B-it/v1"

POPPLER_PATH      = "Release-25.12.0-0/poppler-25.12.0/Library/bin"

client = OpenAI(
    api_key="sk-abc123DEF456ghi789JKL012mno345PQR678stu901VWX234yz",        # 本地部署不需要真實 key
    base_url=VLLM_LLM_API_BASE2
)

# =========================
# 工具函式
# =========================
def crop_image_region(image_pil: Image.Image, bbox: list, padding: int = 30) -> Image.Image:
    """裁切圖片指定區域，加 padding 避免切太緊"""
    w, h    = image_pil.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return image_pil.crop((x1, y1, x2, y2))


def image_to_base64(image_pil: Image.Image) -> str:
    """PIL Image 轉 base64 字串"""
    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_fields_from_image_region(
    image_pil: Image.Image,
    failed_fields: list
) -> dict:
    """
    單次 VLM 辨識，回傳擷取結果
    重試邏輯由 app.py 的比對結果決定
    """

    field_instructions = {
        "金額大寫中文": "- 金額大寫中文：只輸出中文大寫金額本身，不要包含「新臺幣」前綴",
        "未稅金額":    "- 未稅金額：未含稅的銷售金額（純數字，去除逗號）",
        "稅額":       "- 稅額：營業稅金額（純數字，去除逗號）",
        "合計金額":    "- 合計金額：含稅總計金額（純數字，去除逗號）",
        "年度期間":    "- 年度期間：格式為「民國年份年MM-MM月」，例如「115年03-04月」",
        "發票號碼":    "- 發票號碼：2個英文字母+8個數字，例如「AY83205584」",
        "買方統編":    "- 買方統編：買方的8位數字統一編號",
        "賣方統編":    "- 賣方統編：賣方的8位數字統一編號",
        "買方公司名稱": "- 買方公司名稱：完整買方公司名稱",
        "賣方公司名稱": "- 賣方公司名稱：完整賣方公司名稱",
        "明細項目":    """- 明細項目：每筆包含品名、數量、單價、金額，輸出為 JSON 陣列
  注意：
  - 品名請完整輸出，包含前面的料號數字
  - 品名如果換行請合併成完整品名
  - 數量請保留單位（如 40.0 LT）
  - 單價、金額為純數字（去除逗號）
  - 不要包含合計列、稅額列
  例如：[{{"品名": "414288 ENTEK劑", "數量": "40.0 LT", "單價": "1470", "金額": "58800"}}]"""
    }

    json_template = {}
    for field in failed_fields:
        if field == "明細項目":
            json_template[field] = [{"品名": "<值>", "數量": "<原始文字>", "單價": "<純數字>", "金額": "<純數字>"}]
        else:
            json_template[field] = "<值或null>"

    instructions = "\n".join([
        field_instructions[f] for f in failed_fields if f in field_instructions
    ])

    prompt = f"""這是一張發票圖片，請仔細閱讀圖片內容，擷取以下欄位：

{instructions}

請用以下 JSON 格式回答，找不到的欄位填 null，不要加任何多餘說明：
{json.dumps(json_template, ensure_ascii=False, indent=2)}"""

    try:
        b64_image = image_to_base64(image_pil)

        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL2,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }],
            max_tokens=2048,
            temperature=0.0
        )

        content = (response.choices[0].message.content or "").strip()
        print(f"[VLM保底] 回應:\n{content}\n")

        if not content:
            print(f"⚠️  [VLM保底] 回應為空")
            return {}

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            print(f"⚠️  [VLM保底] 找不到 JSON")
            return {}

        result = json.loads(json_match.group())

        # 數字欄位清理
        for key in ["未稅金額", "稅額", "合計金額"]:
            if result.get(key):
                result[key] = str(result[key]).replace(",", "").strip()

        # 明細項目清理
        items = result.get("明細項目", [])
        if isinstance(items, list):
            for item in items:
                for key in ["單價", "金額"]:
                    if item.get(key):
                        item[key] = str(item[key]).replace(",", "").strip()
            result["明細項目"] = items

        return result

    except json.JSONDecodeError as e:
        print(f"⚠️  [VLM保底] JSON 解析失敗：{e}")
        return {}
    except Exception as e:
        print(f"⚠️  [VLM保底] 呼叫失敗：{e}")
        return {}

# =========================
# VLM 判斷是否有多張發票
# =========================
def detect_multi_invoice(image_input, save_debug_path: str = None) -> dict:
    """
    使用 VLM 判斷圖片中是否包含多張發票
    
    Args:
        image_input:      圖片路徑（str）或 PIL Image
        save_debug_path:  若不為 None，將圖片存到此路徑供 debug 用
    Returns:
        dict: {
            "has_multiple_invoices": bool,
            "invoice_count":         int,
            "confidence":            str,   # high / medium / low
            "reason":                str,
            "raw_response":          str
        }
    """
    # 準備圖片
    if isinstance(image_input, str):
        b64 = image_to_base64(image_input)
    elif isinstance(image_input, Image.Image):
        b64 = image_to_base64(image_input)
        if save_debug_path:
            image_input.save(save_debug_path)
            print(f"[DEBUG] 已儲存圖片：{save_debug_path}")
    else:
        raise ValueError("image_input 必須是圖片路徑或 PIL Image")

    # Prompt
    prompt = """請仔細分析這張圖片，判斷圖片中包含幾張發票（電子發票或紙本發票）。

判斷依據：
- 每張發票通常有獨立的發票號碼（如 YW12345678）
- 每張發票有獨立的買方/賣方資訊
- 每張發票有獨立的金額合計

請用以下 JSON 格式回答，不要加任何多餘的說明：
{
  "invoice_count": <數字>,
  "has_multiple_invoices": <true 或 false>,
  "confidence": "<high 或 medium 或 low>",
  "reason": "<簡短說明判斷原因>"
}"""

    # 呼叫 VLM
    response = client.chat.completions.create(
        model=VLLM_LLM_MODEL2,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        max_tokens=256,
        temperature=0.0   # 判斷性任務用 0，減少隨機性
    )

    raw_text = response.choices[0].message.content.strip()
    print(f"[VLM 回應] {raw_text}")

    # 解析 JSON 回應
    try:
        # 有時 VLM 會在 JSON 外面包 markdown ```json ... ```
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(raw_text)

        result["raw_response"] = raw_text
        return result

    except json.JSONDecodeError:
        # 解析失敗時，嘗試從文字判斷
        has_multiple = any(kw in raw_text for kw in ["多張", "兩張", "2張", "multiple", "more than one"])
        return {
            "has_multiple_invoices": has_multiple,
            "invoice_count":         -1,       # 無法確定
            "confidence":            "low",
            "reason":                "JSON解析失敗，依關鍵字推斷",
            "raw_response":          raw_text
        }

# =========================
# 批次處理資料夾內所有圖片/PDF
# =========================
def process_folder(input_dir: str, output_json: str = "multi_invoice_detection.json"):
    """
    批次處理資料夾內所有 PDF 和圖片，輸出偵測結果
    """
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    results = []

    files = sorted(os.listdir(input_dir))
    for filename in files:
        filepath = os.path.join(input_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        pages = []

        if ext == ".pdf":
            print(f"\n處理 PDF：{filename}")
            pdf_pages = convert_from_path(filepath, dpi=300, poppler_path=POPPLER_PATH)
            pages = [(f"{filename}_page{i+1}", p) for i, p in enumerate(pdf_pages)]

        elif ext in IMAGE_EXTENSIONS:
            print(f"\n處理圖片：{filename}")
            img = Image.open(filepath).convert("RGB")
            pages = [(filename, img)]

        else:
            continue

        for page_name, page_img in pages:
            print(f"  偵測：{page_name}")
            detection = detect_multi_invoice(page_img)
            results.append({
                "file":    page_name,
                "result":  detection
            })

            status = "⚠️  多張發票" if detection.get("has_multiple_invoices") else "✅  單張發票"
            print(f"  → {status}（共 {detection.get('invoice_count')} 張，信心度: {detection.get('confidence')}）")
            print(f"     原因: {detection.get('reason')}")

    # 輸出 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果已儲存：{output_json}")
    return results

# =========================
# 主程式
# =========================
if __name__ == "__main__":
    # 單張圖片測試
    test_image = "file/png_output/page_5_page1.png"
    ext = os.path.splitext(test_image)[1].lower()

    if ext == ".pdf":
        pages = convert_from_path(test_image, dpi=700, poppler_path=POPPLER_PATH)
        for i, page in enumerate(pages):
            print(f"\n===== 第 {i+1} 頁 =====")
            result = detect_multi_invoice(page, save_debug_path=f"jpg_pages/detect_page{i+1}.png")
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = detect_multi_invoice(test_image)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    # 批次處理整個資料夾（取消下方註解）
    # process_folder("file/split_output/scan-00003")