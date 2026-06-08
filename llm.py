import json
import re
from openai import OpenAI

# =========================
# LLM 設定
# =========================

VLLM_LLM_MODEL = "gemma-4-26B-A4B-it"
VLLM_LLM_API_BASE = "http://10.2.5.111:8015/gemma-4-26B-A4B-it/v1"
VLLM_API_KEY      = "sk-abc123DEF456ghi789JKL012mno345PQR678stu901VWX234yz"

client = OpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_LLM_API_BASE
)


def extract_invoice_fields_by_llm(ocr_text: str) -> dict:
    """
    將 OCR 文字丟給 LLM，擷取以下欄位：
    - 年度期間
    - 金額大寫中文
    - 未稅金額
    - 稅額
    - 合計金額
    - 明細項目

    Args:
        ocr_text: OCR 辨識後的純文字
    Returns:
        dict: 擷取結果
    """

    prompt = f"""以下是一張發票的 OCR 辨識文字，每行格式為「[x座標,y座標]文字內容」。
請利用座標資訊輔助判斷版面結構：
- Y座標相近的文字代表在同一行
- X座標較小的文字在左邊，X座標較大的文字在右邊
- 同一行中，品名在左、數量/單價/金額依序在右

請從中擷取以下欄位資訊：

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

2. 金額大寫中文：只輸出中文大寫金額本身，不要包含「新臺幣」前綴

3. 未稅金額：未含稅的銷售金額（純數字）

4. 稅額：營業稅金額（純數字）

5. 合計金額：含稅總計金額（純數字）

6. 明細項目：發票中的品項明細，每筆包含品名、數量、單價、金額，輸出為 JSON 陣列
   - ✅ 利用Y座標判斷同一列的品名、數量、單價、金額
   - ✅ 品名可能跨多行（下一行Y座標較大，但該行沒有數量/單價/金額），請合併成完整品名
     例如：「銅添加劑 SLOTOCOUP CU」（第一行）+ 「141」（下一行無數量單價）→ 「銅添加劑 SLOTOCOUP CU141」
   - ✅ 料號數字可能獨立在品名上方，請將料號與品名合併
     例如：「414288」（上一行）+ 「ENTEK PLUS HT RB1促進成膜劑」→ 「414288 ENTEK PLUS HT RB1促進成膜劑」
   - 數量可能含單位（如 "40.0 LT"），請完整保留原始文字
   - 單價、金額為純數字（去除逗號）
   - 只擷取實際品項明細列，不包含合計列、稅額列
   - 找不到的欄位填 null
   例如：[{{"品名": "414288 ENTEK PLUS HT RB1促進成膜劑", "數量": "40.0 LT", "單價": "1470", "金額": "58800"}}]

請用以下 JSON 格式回答，找不到的欄位填 null，不要加任何多餘說明：
{{
  "年度期間": "<值或null>",
  "金額大寫中文": "<值或null>",
  "未稅金額": "<純數字或null>",
  "稅額": "<純數字或null>",
  "合計金額": "<純數字或null>",
  "明細項目": [
    {{"品名": "<值或null>", "數量": "<原始文字或null>", "單價": "<純數字或null>", "金額": "<純數字或null>"}}
  ]
}}

OCR 文字如下：
---
{ocr_text}
---"""

    try:
        print(f"[LLM] 開始呼叫 LLM，模型：{VLLM_LLM_MODEL}")

        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=3072,
            temperature=0.0
        )

        content = response.choices[0].message.content
        print(f"[LLM] 回應內容:\n{content}\n")

        if content is None:
            print(f"⚠️  LLM 回應為 None，完整 response：{response}")
            return _empty_llm_result()

        raw_text = content.strip()

        # 解析 JSON
        try:
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                print(f"⚠️  LLM 回應中找不到 JSON 格式，原始回應：{raw_text}")
                result = _empty_llm_result()
                result["raw_response"] = raw_text
                return result

            # 數字欄位清理（去除逗號）
            for key in ["未稅金額", "稅額", "合計金額"]:
                if result.get(key):
                    result[key] = str(result[key]).replace(",", "").strip()

            # 明細項目數字欄位清理
            detail_items = result.get("明細項目", [])
            if isinstance(detail_items, list):
                for item in detail_items:
                    for key in ["單價", "金額"]:
                        if item.get(key):
                            item[key] = str(item[key]).replace(",", "").strip()
                result["明細項目"] = detail_items
            else:
                result["明細項目"] = []

            result["raw_response"] = raw_text
            print(f"[LLM] 解析結果：{result}")
            return result

        except json.JSONDecodeError as e:
            print(f"⚠️  LLM JSON 解析失敗：{e}，原始回應：{raw_text}")
            result = _empty_llm_result()
            result["raw_response"] = raw_text
            return result

    except Exception as e:
        print(f"⚠️  LLM 呼叫失敗：{e}")
        return _empty_llm_result()


def _empty_llm_result() -> dict:
    """回傳空的 LLM 結果"""
    return {
        "年度期間":    None,
        "金額大寫中文": None,
        "未稅金額":    None,
        "稅額":       None,
        "合計金額":    None,
        "明細項目":    [],
        "raw_response": None
    }

def reextract_specific_fields(ocr_text: str, failed_fields: list) -> dict:
    """
    針對比對失敗的欄位，重新請 LLM 擷取
    Args:
        ocr_text     : OCR 辨識後的文字（含座標）
        failed_fields: 需要重新擷取的欄位清單
    Returns:
        dict: 只包含 failed_fields 的重新擷取結果
    """

    # 各欄位的說明
    field_instructions = {
        "年度期間": """- 年度期間：格式為「民國年份年MM-MM月」，例如「115年03-04月」
   - 若只有開立日期，西元年 - 1911 = 民國年，月份依雙月制推算""",

        "金額大寫中文": """- 金額大寫中文：只輸出中文大寫金額本身，不要包含「新臺幣」前綴""",

        "未稅金額": """- 未稅金額：未含稅的銷售金額（純數字）""",

        "稅額": """- 稅額：營業稅金額（純數字）""",

        "合計金額": """- 合計金額：含稅總計金額（純數字）""",

        "明細項目": """- 明細項目：每筆包含品名、數量、單價、金額，輸出為 JSON 陣列
   - 利用Y座標判斷同一列的品名、數量、單價、金額
   - 品名可能跨多行，請合併成完整品名
   - 料號數字可能獨立在品名上方，請將料號與品名合併
   - 數量可能含單位（如 "40.0 LT"），請完整保留
   - 單價、金額為純數字（去除逗號）
   - 只擷取實際品項明細列，不包含合計列、稅額列""",

      # ✅ 新增 OCR+Regex 欄位
        "發票號碼":    """- 發票號碼：2個英文字母 + 8個數字，例如「BK03970041」
   - 注意可能因換行被切斷，請從上下文還原完整號碼""",
        "買方統編":    """- 買方統編（買方統一編號）：8位數字
   - 通常在「買方」或「買」字附近，可能因換行與公司名稱分開""",
        "賣方統編":    """- 賣方統編（賣方統一編號）：8位數字
   - 通常在發票底部或賣方公司名稱附近""",
        "買方公司名稱": """- 買方公司名稱：完整公司名稱
   - 通常在「買方」或「方:」後面，可能因換行被切斷，請還原完整名稱""",
        "賣方公司名稱": """- 賣方公司名稱：完整公司名稱
   - 通常在發票抬頭或底部，可能因換行被切斷，請還原完整名稱""",
    }

    # 只針對失敗欄位
    instructions = "\n".join([
        field_instructions[f] for f in failed_fields if f in field_instructions
    ])

    # 建立 JSON 格式範例
    json_template = {}
    for field in failed_fields:
        if field == "明細項目":
            json_template[field] = [{"品名": "<值或null>", "數量": "<原始文字或null>", "單價": "<純數字或null>", "金額": "<純數字或null>"}]
        else:
            json_template[field] = "<值或null>"

    prompt = f"""以下是一張發票的 OCR 辨識文字，每行格式為「[x座標,y座標]文字內容」。
請利用座標資訊輔助判斷版面結構：
- Y座標相近的文字代表在同一行
- X座標較小在左，X座標較大在右

請只擷取以下欄位（其他欄位不需要）：
{instructions}

請用以下 JSON 格式回答，找不到的欄位填 null，不要加任何多餘說明：
{json.dumps(json_template, ensure_ascii=False, indent=2)}

OCR 文字如下：
---
{ocr_text}
---"""

    try:
        print(f"[LLM重試] 針對欄位重新擷取: {failed_fields}")

        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0
        )

        content = response.choices[0].message.content
        if not content:
            print(f"⚠️  [LLM重試] 回應為空")
            return {}

        raw_text = content.strip()
        print(f"[LLM重試] 回應:\n{raw_text}\n")

        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not json_match:
            print(f"⚠️  [LLM重試] 找不到 JSON")
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

        return result

    except json.JSONDecodeError as e:
        print(f"⚠️  [LLM重試] JSON 解析失敗：{e}")
        return {}
    except Exception as e:
        print(f"⚠️  [LLM重試] 呼叫失敗：{e}")
        return {}
    

def locate_field_region_by_llm(ocr_text_with_position: str, failed_fields: list) -> dict:
    """
    讓 LLM 根據 OCR 座標文字，推算失敗欄位在圖片中的大概區域（bounding box）
    回傳格式：{"x1": int, "y1": int, "x2": int, "y2": int}
    座標單位與 OCR 輸出的 [x, y] 座標一致，並已含擴展邊界
    """

    field_desc = {
        "明細項目":    "發票明細表格區域，包含品名、數量、單價、金額欄位的所有列（含表頭與最後一列）",
        "金額大寫中文": "中文大寫金額所在區域",
        "未稅金額":    "未稅金額數字所在區域",
        "稅額":       "稅額數字所在區域",
        "合計金額":    "合計金額數字所在區域",
        "發票號碼":    "發票號碼（英文+數字）所在區域",
        "買方公司名稱": "買方公司名稱所在區域",
        "賣方公司名稱": "賣方公司名稱所在區域",
        "買方統編":    "買方統一編號（8位數字）所在區域",
        "賣方統編":    "賣方統一編號（8位數字）所在區域",
    }

    targets = "\n".join([
        f"- {f}：{field_desc.get(f, f)}" for f in failed_fields
    ])

    prompt = f"""以下是一張發票的 OCR 辨識文字，每行格式為「[x座標,y座標]文字內容」。

請判斷下列欄位在圖片中的位置，回傳一個能完整涵蓋所有欄位的矩形區域（bounding box）：

目標欄位：
{targets}

重要注意事項：
- OCR 座標只標記文字的起始位置，實際內容（尤其是表格右側欄位）可能延伸更遠
- 請在找到的座標範圍基礎上，四個方向各額外擴展約 200 像素的安全邊界
- 寧可框大一點，也不要切到內容

請回傳以下 JSON 格式（座標單位與 OCR 座標相同，已含安全邊界）：
{{
  "x1": <最左邊x座標 再往左擴200，整數>,
  "y1": <最上方y座標 再往上擴200，整數>,
  "x2": <最右邊x座標 再往右擴200，整數>,
  "y2": <最下方y座標 再往下擴200，整數>,
  "reason": "<簡短說明判斷依據>"
}}

只回傳 JSON，不要加任何說明。

OCR 文字如下：
---
{ocr_text_with_position}
---"""

    try:
        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0
        )

        content = (response.choices[0].message.content or "").strip()
        print(f"[LLM區域定位] 回應：{content}")

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            print("⚠️  [LLM區域定位] 找不到 JSON")
            return {}

        result = json.loads(json_match.group())

        # 確保都是整數
        for key in ["x1", "y1", "x2", "y2"]:
            if key in result:
                result[key] = int(result[key])

        # ✅ 程式側再加一層保險擴展，以防 LLM 沒有確實擴展
        EXTRA_MARGIN = 150
        result["x1"] = max(0, result.get("x1", 0) - EXTRA_MARGIN)
        result["y1"] = max(0, result.get("y1", 0) - EXTRA_MARGIN)
        result["x2"] = result.get("x2", 0) + EXTRA_MARGIN
        result["y2"] = result.get("y2", 0) + EXTRA_MARGIN

        print(f"[LLM區域定位] 擴展後座標：x1={result['x1']} y1={result['y1']} x2={result['x2']} y2={result['y2']}")
        return result

    except Exception as e:
        print(f"⚠️  [LLM區域定位] 失敗：{e}")
        return {}


# # =========================
# # LLM 擷取發票欄位
# # =========================
# def extract_invoice_fields_by_llm(ocr_text: str) -> dict:
#     """
#     將 OCR 文字丟給 LLM，擷取以下欄位：
#     - 年度期間
#     - 金額大寫中文
#     - 未稅金額
#     - 稅額
#     - 合計金額

#     Args:
#         ocr_text: OCR 辨識後的純文字
#     Returns:
#         dict: 擷取結果
#     """

#     prompt = f"""以下是一張發票的 OCR 辨識文字，請從中擷取以下欄位資訊：

# 1. 年度期間：發票的年度與月份期間，格式為「民國年份年MM-MM月」，例如「115年03-04月」
#    - 情況A：發票直接標示期間，例如「115年03-04月」→ 直接使用
#    - 情況B：發票只有開立日期（西元或民國），請依以下規則推算期間：
#      * 西元年轉民國年：西元年 - 1911，例如 2026 → 115
#      * 月份對應雙月期間（發票為雙月制）：
#        01月或02月 → 01-02月
#        03月或04月 → 03-04月
#        05月或06月 → 05-06月
#        07月或08月 → 07-08月
#        09月或10月 → 09-10月
#        11月或12月 → 11-12月
#      * 例如：2026-03-17 → 115年，03月 → 03-04月 → 輸出「115年03-04月」

# 2. 金額大寫中文：發票金額的中文大寫，✅ 只輸出中文大寫金額本身，不要包含「新臺幣」前綴
#    例如：「新臺幣陸拾捌元整」→ 輸出「陸拾捌元整」
# 3. 未稅金額：未含稅的銷售金額（純數字）
# 4. 稅額：營業稅金額（純數字）
# 5. 合計金額：含稅總計金額（純數字）

# 6. 明細項目：發票中的品項明細，每筆包含品名、數量、單價、金額，輸出為 JSON 陣列
#    例如：[{{"品名": "KA237 5*33.0*2350", "數量": 2000, "單價": 19, "金額": 38000}}]
#    - 數量、單價、金額皆為純數字
#    - 找不到的欄位填 null

# 請用以下 JSON 格式回答，找不到的欄位填 null，不要加任何多餘說明：
# {{
#   "年度期間": "<值或null>",
#   "金額大寫中文": "<值或null>",
#   "未稅金額": "<純數字或null>",
#   "稅額": "<純數字或null>",
#   "合計金額": "<純數字或null>",
#   "明細項目": [
#     {{"品名": "<值或null>", "數量": "<純數字或null>", "單價": "<純數字或null>", "金額": "<純數字或null>"}}
#   ]
# }}

# OCR 文字如下：
# ---
# {ocr_text}
# ---"""

#     response = client.chat.completions.create(
#         model=VLLM_LLM_MODEL,
#         messages=[
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=1024,
#         temperature=0.0
#     )
#     # ✅ 加入 None 保護
#     content = response.choices[0].message.content
#     if content is None:
#         print(f"[LLM] 回應內容為 None，印出完整 response 供 debug：")
#         print(response)
#         return {
#             "年度期間":    None,
#             "金額大寫中文": None,
#             "未稅金額":    None,
#             "稅額":       None,
#             "合計金額":    None,
#             "raw_response": None
#         }
    
#     # raw_text = response.choices[0].message.content.strip()
#     raw_text = content.strip()
#     print(f"[LLM 回應] {raw_text}")

#     # 解析 JSON
#     try:
#         json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
#         if json_match:
#             result = json.loads(json_match.group())
#         else:
#             result = json.loads(raw_text)

#         # 數字欄位清理（去除逗號）
#         for key in ["未稅金額", "稅額", "合計金額"]:
#             if result.get(key):
#                 result[key] = str(result[key]).replace(",", "").strip()

#         result["raw_response"] = raw_text
#         return result

#     except json.JSONDecodeError:
#         print(f"[LLM] JSON 解析失敗，回傳原始文字")
#         return {
#             "年度期間":    None,
#             "金額大寫中文": None,
#             "未稅金額":    None,
#             "稅額":       None,
#             "合計金額":    None,
#             "明細項目":    [],
#             "raw_response": raw_text
#         }