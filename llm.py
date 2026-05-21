import json
import re
from openai import OpenAI

# =========================
# LLM 設定
# =========================
VLLM_LLM_MODEL    = "gemma-3-27b-it"
VLLM_LLM_API_BASE = "http://10.2.5.111:8015/gemma-3-27b-it/v1"
VLLM_API_KEY      = "sk-abc123DEF456ghi789JKL012mno345PQR678stu901VWX234yz"

VLLM_LLM_MODEL = "gpt-oss-120b"
VLLM_LLM_API_BASE = "http://10.2.5.111:8015/gpt-oss-120b/v1"
VLLM_API_KEY      = "sk-abc123DEF456ghi789JKL012mno345PQR678stu901VWX234yz"

client = OpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_LLM_API_BASE
)

# =========================
# LLM 擷取發票欄位
# =========================
def extract_invoice_fields_by_llm(ocr_text: str) -> dict:
    """
    將 OCR 文字丟給 LLM，擷取以下欄位：
    - 年度期間
    - 金額大寫中文
    - 未稅金額
    - 稅額
    - 合計金額

    Args:
        ocr_text: OCR 辨識後的純文字
    Returns:
        dict: 擷取結果
    """

    prompt = f"""以下是一張發票的 OCR 辨識文字，請從中擷取以下欄位資訊：

1. 年度期間：發票的年度與月份期間，格式為「民國年份年MM-MM月」，例如「115年03-04月」
   - 情況A：發票直接標示期間，例如「115年03-04月」→ 直接使用
   - 情況B：發票只有開立日期（西元或民國），請依以下規則推算期間：
     * 西元年轉民國年：西元年 - 1911，例如 2026 → 115
     * 月份對應雙月期間（發票為雙月制）：
       01月或02月 → 01-02月
       03月或04月 → 03-04月
       05月或06月 → 05-06月
       07月或08月 → 07-08月
       09月或10月 → 09-10月
       11月或12月 → 11-12月
     * 例如：2026-03-17 → 115年，03月 → 03-04月 → 輸出「115年03-04月」

2. 金額大寫中文：發票金額的中文大寫，✅ 只輸出中文大寫金額本身，不要包含「新臺幣」前綴
   例如：「新臺幣陸拾捌元整」→ 輸出「陸拾捌元整」
3. 未稅金額：未含稅的銷售金額（純數字）
4. 稅額：營業稅金額（純數字）
5. 合計金額：含稅總計金額（純數字）

請用以下 JSON 格式回答，找不到的欄位填 null，不要加任何多餘說明：
{{
  "年度期間": "<值或null>",
  "金額大寫中文": "<值或null>",
  "未稅金額": "<純數字或null>",
  "稅額": "<純數字或null>",
  "合計金額": "<純數字或null>"
}}

OCR 文字如下：
---
{ocr_text}
---"""

    response = client.chat.completions.create(
        model=VLLM_LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.0
    )
    # ✅ 加入 None 保護
    content = response.choices[0].message.content
    if content is None:
        print(f"[LLM] 回應內容為 None，印出完整 response 供 debug：")
        print(response)
        return {
            "年度期間":    None,
            "金額大寫中文": None,
            "未稅金額":    None,
            "稅額":       None,
            "合計金額":    None,
            "raw_response": None
        }
    
    # raw_text = response.choices[0].message.content.strip()
    raw_text = content.strip()
    print(f"[LLM 回應] {raw_text}")

    # 解析 JSON
    try:
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(raw_text)

        # 數字欄位清理（去除逗號）
        for key in ["未稅金額", "稅額", "合計金額"]:
            if result.get(key):
                result[key] = str(result[key]).replace(",", "").strip()

        result["raw_response"] = raw_text
        return result

    except json.JSONDecodeError:
        print(f"[LLM] JSON 解析失敗，回傳原始文字")
        return {
            "年度期間":    None,
            "金額大寫中文": None,
            "未稅金額":    None,
            "稅額":       None,
            "合計金額":    None,
            "raw_response": raw_text
        }