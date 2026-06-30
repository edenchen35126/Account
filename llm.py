import json
import re
from openai import OpenAI

# =========================
# LLM 設定
# =========================

VLLM_LLM_MODEL = "gemma-4-26B-A4B-it"
VLLM_LLM_API_BASE = "http://10.2.5.22:8190/v1"


VLLM_API_KEY      = "sk-Wz-SOJu0vl6_0HHlVuaRXQ"

client = OpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_LLM_API_BASE
)


def _to_confidence_01(value):
    """將模型回傳的信心值轉成 0~1 浮點數，失敗回傳 None。"""
    if value is None:
        return None
    try:
        v = float(value)
        if 0.0 <= v <= 1.0:
            return round(v, 4)
    except (TypeError, ValueError):
        return None
    return None


def _normalize_field_confidence_map(raw_map, allowed_fields: list) -> dict:
    """清洗欄位信心值，只保留允許欄位且數值在 0~1。"""
    if not isinstance(raw_map, dict):
        return {}

    cleaned = {}
    for f in allowed_fields:
        if f in raw_map:
            conf = _to_confidence_01(raw_map.get(f))
            if conf is not None:
                cleaned[f] = conf
    return cleaned


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
2. 發票日期：發票上的開立日期
   - 可能是民國格式（如「115年03月17日」）或西元格式（如「2026/03/17」或「2026-03-17」）
   - 直接輸出原始格式，不需轉換
   - 找不到填 null
3. 金額大寫中文：只輸出中文大寫金額本身，不要包含「新臺幣」前綴
4. 未稅金額：未含稅的銷售金額（純數字）
5. 稅額：營業稅金額（純數字）
6. 合計金額：含稅總計金額（純數字）
7. 明細項目：發票中的品項明細，每筆包含品名、數量、單價、金額，輸出為 JSON 陣列
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
  "發票日期": "<值或null>",
  "金額大寫中文": "<值或null>",
  "未稅金額": "<純數字或null>",
  "稅額": "<純數字或null>",
  "合計金額": "<純數字或null>",
  "明細項目": [
    {{"品名": "<值或null>", "數量": "<原始文字或null>", "單價": "<純數字或null>", "金額": "<純數字或null>"}}
    ],
    "欄位信心值": {{
        "年度期間": "<0~1或null>",
        "發票日期": "<0~1或null>",
        "金額大寫中文": "<0~1或null>",
        "未稅金額": "<0~1或null>",
        "稅額": "<0~1或null>",
        "合計金額": "<0~1或null>",
        "明細項目": "<0~1或null>"
    }}
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

            result["__field_confidence__"] = _normalize_field_confidence_map(
                result.get("欄位信心值"),
                ["年度期間", "發票日期", "金額大寫中文", "未稅金額", "稅額", "合計金額", "明細項目"]
            )
            result.pop("欄位信心值", None)

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
        "發票日期": None,
        "金額大寫中文": None,
        "未稅金額":    None,
        "稅額":       None,
        "合計金額":    None,
        "明細項目":    [],
        "__field_confidence__": {},
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
        "發票日期": """- 發票日期：發票上的開立日期（民國或西元格式，直接輸出原始格式）""",

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

        # ✅ OCR+Regex 欄位
        "發票號碼":    """- 發票號碼：2個英文字母 + 8個數字，例如「BK03970041」
   - 注意可能因換行被切斷，請從上下文還原完整號碼""",
        "買方統編":    """- 買方統編（買方統一編號）：8位數字
   - 通常在「買方」或「買」字附近，可能因換行與公司名稱分開""",
        "賣方統編":    """- 賣方統編（賣方統一編號）：8位數字
   - 請優先參考 Y 座標位置：賣方資訊通常位於發票底部、賣方章戳附近、或賣方公司名稱附近""",
        "買方公司名稱": """- 買方公司名稱：完整公司名稱
   - 通常在「買方」或「方:」後面，可能因換行被切斷，請還原完整名稱""",
   #   - 請優先參考 Y 座標位置：賣方資訊通常位於發票底部、賣方章戳附近 
   #   - 通常在發票抬頭或底部，可能因換行被切斷，請還原完整名稱 
   #   - 若 OCR 只出現殘缺片段，例如「股份有限公」、「有限公司」等，且無法確認完整名稱，請輸出 null
        "賣方公司名稱": """- 賣方公司名稱：完整賣方公司名稱
   - 只能從「賣方資訊區」擷取，不可從買方欄位推測或補值
   - 賣方資訊區通常位於：
     1. 發票底部
     2. 「營業人蓋用統一發票專用章」附近
     3. 賣方章戳附近
     4. 賣方統一編號附近
   - 若同一張發票同時出現買方公司名稱與賣方公司名稱，請依 Y 座標與鄰近文字判斷：
     - 買方公司名稱通常靠近「買受人」、「買方」、「統一編號」等買方欄位
     - 賣方公司名稱通常靠近底部章戳、賣方統編、發票專用章
   - 嚴格禁止將「買方公司名稱」填入「賣方公司名稱」
   - 嚴格禁止因為賣方公司名稱辨識不完整，就使用買方公司名稱補上
   - 若只找到買方公司名稱，但沒有找到明確位於賣方區域的公司名稱，請輸出 null
   - 若 OCR 只出現殘缺片段，例如「股份有限公」、「有限公司」、「電子股份有限公」等，且無法從賣方區域確認完整名稱，請輸出 null
   - 不可根據常識、公司名稱完整度、買方名稱或上下文自行推測賣方公司名稱""",
        "營業稅稅別判斷": """- 營業稅稅別判斷：判斷發票上勾選的是「應稅」、「零稅率」或「免稅」
   - 台灣統一發票版型由左到右：應稅 | 應稅勾選框 | 零稅率 | 零稅率勾選框 | 免稅 | 免稅勾選框
   - 勾選框在該選項文字的右邊，不是左邊
   - 若看到「應稅  √  零稅率」，代表勾選的是「應稅」
   - 只輸出：應稅 或 零稅率 或 免稅""",
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
        elif field == "營業稅稅別判斷":
            json_template[field] = "<應稅 或 零稅率 或 免稅 或 null>"
        else:
            json_template[field] = "<值或null>"

    # 要求模型同步回傳每欄位的主觀信心值（0~1）
    json_template["欄位信心值"] = {field: "<0~1或null>" for field in failed_fields}

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

        result["__field_confidence__"] = _normalize_field_confidence_map(
            result.get("欄位信心值"),
            failed_fields
        )
        result.pop("欄位信心值", None)

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
        "營業稅稅別判斷": "營業稅稅別勾選區域（應稅/零稅率/免稅）",
    }

    targets = "\n".join([
        f"- {f}：{field_desc.get(f, f)}" for f in failed_fields
    ])

    seller_tax_hint = ""
    if "賣方統編" in failed_fields:
        seller_tax_hint = """
【賣方統編定位特別規則】
- 賣方統編通常在發票下半部，常靠近「營業人蓋章」、「負責人」、「TEL」、「地址」、「統一編號」等文字。
- 請避免抓到上半部「買受人/買方統編」區域。
- 若同時看到買方統編與賣方統編，請優先回傳靠近頁面下半部的賣方統編區域。
"""

    prompt = f"""以下是一張發票的 OCR 辨識文字，每行格式為「[x座標,y座標]文字內容」。

請判斷下列欄位在圖片中的位置，回傳一個能完整涵蓋所有欄位的矩形區域（bounding box）：

目標欄位：
{targets}

{seller_tax_hint}

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


def determine_tax_type_by_llm(ocr_text: str) -> dict:
    """
    讓 LLM 判斷發票上是否有勾選「應稅」、「零稅率」或「免稅」其中一項。

    Args:
        ocr_text: OCR 辨識後的純文字（含座標）
    Returns:
        dict: {
          "稅別": "應稅" / "零稅率" / "免稅" / None（找不到勾選）,
          "已勾選": True / False,
          "判斷依據": "..."
        }
    """
    prompt = f"""以下是一張發票的 OCR 辨識文字，每行格式為「[x座標,y座標]文字內容」。

請判斷這張發票上是否有勾選「應稅」、「零稅率」或「免稅」其中一項。

判斷方式：
- 發票上通常有勾選框，標示「應稅」、「零稅率」或「免稅」
- 若 OCR 文字中出現該字樣，且附近有「✓」「V」「■」「☑」「●」等勾選/選取符號，即視為已勾選
- 若只出現其中一項字樣（無其他候選項並列），也視為已勾選
- 若三種選項都出現但無法判斷哪個被勾選，請回傳 null
- 若完全找不到任何相關文字，請回傳 null

請只回傳以下 JSON 格式，不要加任何說明：
{{
  "稅別": "<應稅 或 零稅率 或 免稅 或 null>",
  "已勾選": <true 或 false>,
    "判斷依據": "<簡短說明>",
    "信心值": "<0~1或null>"
}}

OCR 文字如下：
---
{ocr_text}
---"""

    try:
        print(f"[LLM稅別判斷] 開始判斷稅別勾選狀態...")
        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0
        )
        content = (response.choices[0].message.content or "").strip()
        print(f"[LLM稅別判斷] 回應：{content}")

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            print("⚠️  [LLM稅別判斷] 找不到 JSON，視為未找到勾選")
            return {"稅別": None, "已勾選": False, "判斷依據": "LLM回應解析失敗"}

        result   = json.loads(json_match.group())
        tax_type = result.get("稅別")
        found    = bool(result.get("已勾選", False))
        reason   = result.get("判斷依據", "")

        # 稅別只接受合法值，否則視為找不到
        VALID_TAX_TYPES = ["應稅", "零稅率", "免稅"]
        if tax_type not in VALID_TAX_TYPES:
            print(f"⚠️  [LLM稅別判斷] 未知稅別 [{tax_type}]，視為未找到勾選")
            return {"稅別": None, "已勾選": False, "判斷依據": reason or f"未知稅別：{tax_type}"}

        conf = _to_confidence_01(result.get("信心值"))
        print(f"[LLM稅別判斷] 結果：{tax_type}，已勾選={found}（{reason}）")
        return {"稅別": tax_type, "已勾選": found, "判斷依據": reason, "信心值": conf}

    except Exception as e:
        print(f"⚠️  [LLM稅別判斷] 失敗：{e}")
        return {"稅別": None, "已勾選": False, "判斷依據": f"LLM呼叫失敗：{e}"}


# =========================
# ✅ 中文大寫金額轉整數（純程式碼，不依賴 LLM）
# =========================
def chinese_amount_to_int(s: str) -> int | None:
    """
    將中文大寫金額轉成整數。
    核心規則：「零」是佔位符，「零佰」「零拾」中的零直接跳過，不參與乘法。

    範例：
      「零億零仟零佰參拾柒萬玖仟玖佰柒拾肆元整」→ 379974
      「零仟零佰零拾參萬玖仟玖佰零拾零元整」      → 39900
      「參拾柒萬玖仟玖佰柒拾肆」                  → 379974
    """
    if not s:
        return None

    s = re.sub(r'\s+', '', s)
    s = re.sub(r'^新臺幣', '', s)
    s = re.sub(r'元整$|元$', '', s)

    digit_map = {
        '零': 0, '壹': 1, '貳': 2, '參': 3, '肆': 4,
        '伍': 5, '陸': 6, '柒': 7, '捌': 8, '玖': 9,
        # 也接受一般國字（OCR 可能辨識成這些）
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9,
    }
    unit_map = {
        '仟': 1000, '千': 1000,
        '佰': 100,  '百': 100,
        '拾': 10,   '十': 10,
    }

    def parse_segment(seg: str) -> int:
        """
        解析一段不超過 9999 的中文數字。
        規則：零後面跟著單位字（佰/拾等）→ 整組跳過（佔位零不計入數值）。
        """
        val = 0
        i = 0
        while i < len(seg):
            ch = seg[i]

            # 遇到零：往後跳過所有緊接的單位字（零仟/零佰/零拾）
            if ch == '零':
                i += 1
                while i < len(seg) and seg[i] in unit_map:
                    i += 1  # 跳過緊接在零後面的單位字
                continue

            if ch in digit_map:
                digit = digit_map[ch]
                # 看下一個字是不是單位
                if i + 1 < len(seg) and seg[i + 1] in unit_map:
                    unit = unit_map[seg[i + 1]]
                    val += digit * unit
                    i += 2  # 跳過數字和單位
                else:
                    # 個位數（後面沒有單位）
                    val += digit
                    i += 1
            elif ch in unit_map:
                # 單位前沒有數字（如「拾」獨立出現代表 10）
                val += unit_map[ch]
                i += 1
            else:
                i += 1

        return val

    total     = 0
    remaining = s

    # 處理億段
    yi_match = re.match(r'^(.+?)億(.*)$', remaining)
    if yi_match:
        total    += parse_segment(yi_match.group(1)) * 100_000_000
        remaining = yi_match.group(2)

    # 處理萬段
    wan_match = re.match(r'^(.+?)萬(.*)$', remaining)
    if wan_match:
        total    += parse_segment(wan_match.group(1)) * 10_000
        remaining = wan_match.group(2)

    # 處理元段（個位到千位）
    total += parse_segment(remaining)

    return total if total > 0 else None


# =========================
# ✅ 中文大寫金額比對（程式碼優先，LLM 後備）
# =========================
def compare_chinese_amount_meaning_by_llm(ocr_chinese: str, total_amount) -> tuple:
    """
    比對 OCR 中文大寫金額與合計金額數字是否相符。
    優先用程式碼轉換比對，LLM 只作後備（避免 LLM 解析中文大寫出錯）。

    Args:
        ocr_chinese:  OCR 擷取到的中文大寫金額字串
        total_amount: 計算出的合計金額（數字）
    Returns:
        tuple: (是否相符: bool, 判斷說明: str)
    """
    if not ocr_chinese:
        return False, "OCR未擷取到中文大寫金額"
    if total_amount is None:
        return False, "合計金額為空，無法比對"

    # ✅ 第一步：程式碼直接轉換比對
    try:
        converted = chinese_amount_to_int(ocr_chinese)
        std_int   = int(float(str(total_amount).replace(",", "")))
        print(f"[中文金額比對] 程式碼轉換：OCR={ocr_chinese!r} → {converted}，標準答案={std_int}")

        def build_repair_candidates(raw: str, target: int) -> list[str]:
            """
            針對手寫發票「中文大寫定位格」常見噪音建立修復候選。
            典型問題：固定單位字被 OCR 合進內容，造成數值被放大（如 10607200）。
            """
            base = re.sub(r'\s+', '', str(raw))
            base = re.sub(r'^新臺幣', '', base)
            base = re.sub(r'元整$|元$', '元', base)

            candidates = [base]

            # 常見誤植前綴（定位格左側空白常被誤判成「壹仟」等）
            for prefix in ["壹仟", "一千", "壹千", "壹佰", "一百"]:
                if base.startswith(prefix):
                    candidates.append(base[len(prefix):])

            # 小金額常見誤植：把「X萬」誤成「X拾萬 / X佰萬 / X仟萬」
            if target < 100_000:
                candidates.append(re.sub(r'([壹貳參肆伍陸柒捌玖一二三四五六七八九])拾萬', r'\1萬', base))
                candidates.append(re.sub(r'([壹貳參肆伍陸柒捌玖一二三四五六七八九])佰萬', r'\1萬', base))
                candidates.append(re.sub(r'([壹貳參肆伍陸柒捌玖一二三四五六七八九])仟萬', r'\1萬', base))

                # 若「萬」前面混入多餘單位，取萬前最後一個數字字元（例：壹仟陸拾萬 -> 陸萬）
                if '萬' in base:
                    pre, post = base.split('萬', 1)
                    numerals = re.findall(r'[壹貳參肆伍陸柒捌玖一二三四五六七八九零]', pre)
                    if numerals:
                        candidates.append(f"{numerals[-1]}萬{post}")

            # 去重且維持順序
            dedup = []
            seen = set()
            for c in candidates:
                if not c or c in seen:
                    continue
                seen.add(c)
                dedup.append(c)
            return dedup

        def try_repaired_match(raw: str, target: int):
            for cand in build_repair_candidates(raw, target):
                val = chinese_amount_to_int(cand)
                print(f"[中文金額比對][修復候選] {cand!r} -> {val}")
                if val is None:
                    continue
                if val == target:
                    return True, cand, val
                if abs(val - target) / max(target, 1) < 0.01:
                    return True, cand, val
            return False, None, None

        if converted is not None:
            if converted == std_int:
                return True, f"程式碼轉換相符（{converted:,}）"

            # 差距在 1% 以內視為四捨五入誤差（如稅額進位）
            if abs(converted - std_int) / max(std_int, 1) < 0.01:
                return True, f"程式碼轉換近似相符（{converted:,} ≈ {std_int:,}）"

            repaired_ok, repaired_text, repaired_val = try_repaired_match(ocr_chinese, std_int)
            if repaired_ok:
                return True, f"OCR噪音修正後相符（{repaired_text} -> {repaired_val:,}）"

            # 差距明確，直接判不符，不需再問 LLM
            return False, f"中文大寫金額為 {converted:,} 元，與數字金額 {std_int:,} 元不符。"

    except Exception as e:
        print(f"[中文金額比對] 程式碼轉換失敗：{e}，改用 LLM")

    # ✅ 第二步（後備）：LLM 比對（程式碼轉換失敗時才進入）
    prompt = f"""請判斷下列中文大寫金額與數字金額的意思是否相符。

中文大寫金額（OCR辨識）：{ocr_chinese}
數字金額（計算結果）：{total_amount}

判斷規則：
- 忽略「新臺幣」前綴
- 忽略「元整」「元」後綴的差異
- 忽略前綴零（如「零仟零佰」）
- 只要表達的數值相同即算相符
- 例如：「陸萬壹仟元整」與 61000 相符

請只回傳以下 JSON 格式，不要加任何說明：
{{
  "是否相符": true 或 false,
  "判斷說明": "<簡短說明>"
}}"""

    try:
        print(f"[中文金額比對] 程式碼轉換失敗，改用 LLM 比對")
        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0
        )
        content = (response.choices[0].message.content or "").strip()
        print(f"[中文金額比對] LLM 回應：{content}")

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            return False, "LLM回應解析失敗"

        result  = json.loads(json_match.group())
        matched = bool(result.get("是否相符", False))
        reason  = result.get("判斷說明", "")
        print(f"[中文金額比對] LLM 結果：{matched}（{reason}）")
        return matched, reason

    except Exception as e:
        print(f"⚠️  [中文金額比對] LLM 呼叫失敗：{e}")
        return False, f"LLM呼叫失敗：{e}"


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