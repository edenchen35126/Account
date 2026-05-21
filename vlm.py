import os
import base64
import json
import re
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image
import numpy as np

# =========================
# VLM 設定
# =========================
VLLM_LLM_MODEL2 = "gemma-3-27b-it"
VLLM_LLM_API_BASE2 = "http://10.2.5.111:8015/gemma-3-27b-it/v1"

POPPLER_PATH      = "Release-25.12.0-0/poppler-25.12.0/Library/bin"

client = OpenAI(
    api_key="sk-abc123DEF456ghi789JKL012mno345PQR678stu901VWX234yz",        # 本地部署不需要真實 key
    base_url=VLLM_LLM_API_BASE2
)

# =========================
# 工具函式
# =========================
def image_to_base64(image_path: str) -> str:
    """將圖片轉成 base64 字串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def pil_to_base64(pil_image: Image.Image) -> str:
    """將 PIL Image 轉成 base64 字串"""
    import io
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

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
        b64 = pil_to_base64(image_input)
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
        pages = convert_from_path(test_image, dpi=300, poppler_path=POPPLER_PATH)
        for i, page in enumerate(pages):
            print(f"\n===== 第 {i+1} 頁 =====")
            result = detect_multi_invoice(page, save_debug_path=f"jpg_pages/detect_page{i+1}.png")
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = detect_multi_invoice(test_image)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    # 批次處理整個資料夾（取消下方註解）
    # process_folder("file/split_output/scan-00003")