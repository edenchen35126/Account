import os
import base64
import json
import re
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from io import BytesIO

from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()

# =========================
# VLM 設定
# =========================
VLLM_LLM_MODEL2 = "utllm-s"
VLLM_LLM_MODEL3 = "utllm-a"
# VLLM_LLM_API_BASE2 = "http://10.2.5.111:8015/gemma-4-26B-A4B-it/v1"

VLLM_LLM_API_BASE2 = "http://mis-4142:8190/v1"

POPPLER_PATH      = "Release-25.12.0-0/poppler-25.12.0/Library/bin"

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

client = OpenAI(
    # api_key="sk-abc123DEF456ghi789JKL012mno345PQR678stu901VWX234yz",        # 本地部署不需要真實 key
    api_key="sk-IMUDNSsGwTMxWIaraQzcyw",
    base_url=VLLM_LLM_API_BASE2
)


def _to_confidence_01(value):
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
    if not isinstance(raw_map, dict):
        return {}
    cleaned = {}
    for f in allowed_fields:
        if f in raw_map:
            conf = _to_confidence_01(raw_map.get(f))
            if conf is not None:
                cleaned[f] = conf
    return cleaned

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


# =========================
# 公司存在性（用模型做保守判斷）
# =========================

_company_existence_cache: dict[str, dict] = {}

def _normalize_company_name_for_check(name: str) -> str:
    if not name:
        return ""
    s = re.sub(r"\s+", "", str(name))
    s = s.replace("臺", "台")
    return s

# from urllib.parse import urlparse
# def verify_company_existence_by_model(company_name: str) -> dict:
#     """
#     用 Tavily 搜尋公司名稱是否存在。

#     判斷規則：
#     - True  : 政府 / 經濟部 / 商工登記等可信來源中，有完整相符公司名稱
#     - None  : 有搜尋到完整公司名稱，但來源不是可信政府/登記來源
#     - False : 搜尋結果中完全沒有完整相符公司名稱，或名稱明顯不像公司名

#     回傳:
#     {
#         "exists": True/False/None,
#         "reason": "...",
#         "source": "...",
#         "matched_url": "...",
#         "matched_title": "..."
#     }
#     """
#     name = (company_name or "").strip()
#     if not name:
#         return {"exists": None, "reason": "empty"}

#     key = _normalize_company_name_for_check(name)
#     cache_key = f"tavily:{key}"

#     if cache_key in _company_existence_cache:
#         return _company_existence_cache[cache_key]

#     trusted_domains = {
#         "findbiz.nat.gov.tw",
#         "data.gcis.nat.gov.tw",
#         "gcis.nat.gov.tw",
#     }

#     supporting_domains = {
#         "104.com.tw",
#         "1111.com.tw",
#         "yes123.com.tw",
#         "findcompany.com.tw",
#         "twincn.com",
#         "iyp.com.tw",
#         "info.technews.tw",
#         "mygov.tw",
#     }

#     def get_domain(url: str) -> str:
#         try:
#             netloc = urlparse(url).netloc.lower()
#             if netloc.startswith("www."):
#                 netloc = netloc[4:]
#             return netloc
#         except Exception:
#             return ""

#     def domain_in(domain: str, allow_domains: set[str]) -> bool:
#         return any(domain == d or domain.endswith("." + d) for d in allow_domains)

#     def normalize_for_match(text: str) -> str:
#         text = text or ""
#         text = re.sub(r"\s+", "", text)
#         text = text.replace("　", "")
#         return text

#     def has_exact_company_name(text: str, target_name: str) -> bool:
#         return normalize_for_match(target_name) in normalize_for_match(text)

#     try:
#         # 先做非常基本的格式檢查，避免 OCR 明顯雜訊直接打搜尋
#         normalized_name = _normalize_company_name_for_check(name)

#         if not normalized_name:
#             result = {
#                 "exists": False,
#                 "reason": "公司名稱清洗後為空",
#                 "source": "format_check",
#             }
#             _company_existence_cache[cache_key] = result
#             return result

#         # 明顯不像台灣公司名稱的先擋掉
#         company_suffixes = (
#             "有限公司",
#             "股份有限公司",
#             "有限合夥",
#             "商行",
#             "企業社",
#             "工作室",
#             "行",
#             "店",
#         )

#         if not any(normalized_name.endswith(suffix) for suffix in company_suffixes):
#             result = {
#                 "exists": False,
#                 "reason": "名稱不像常見台灣公司或商業登記名稱",
#                 "source": "format_check",
#             }
#             _company_existence_cache[cache_key] = result
#             return result

#         queries = [
#             f'"{normalized_name}" 統一編號 公司登記 經濟部',
#             f'"{normalized_name}" 商工登記',
#             f'"{normalized_name}" 公司登記',
#         ]

#         evidence = []
#         seen_urls = set()

#         for query in queries:
#             response = tavily_client.search(
#                 query=query,
#                 search_depth="advanced",
#                 max_results=10,
#                 include_answer=False,
#                 include_raw_content=False,
#             )

#             for item in response.get("results", []):
#                 url = item.get("url", "") or ""
#                 if not url or url in seen_urls:
#                     continue

#                 seen_urls.add(url)

#                 title = item.get("title", "") or ""
#                 content = item.get("content", "") or ""
#                 combined_text = f"{title} {content}"

#                 # 沒有完整公司名的結果先不列入命中
#                 if not has_exact_company_name(combined_text, normalized_name):
#                     continue

#                 domain = get_domain(url)

#                 hit = {
#                     "title": title,
#                     "url": url,
#                     "domain": domain,
#                     "content": content[:500],
#                     "score": item.get("score"),
#                     "query": query,
#                     "is_trusted": domain_in(domain, trusted_domains),
#                     "is_supporting": domain_in(domain, supporting_domains),
#                 }

#                 evidence.append(hit)

#         trusted_hits = [item for item in evidence if item["is_trusted"]]
#         supporting_hits = [item for item in evidence if item["is_supporting"]]
#         other_hits = [
#             item for item in evidence
#             if not item["is_trusted"] and not item["is_supporting"]
#         ]

#         # 1. 政府 / 經濟部 / 商工登記可信來源有完整公司名稱
#         if trusted_hits:
#             hit = trusted_hits[0]
#             result = {
#                 "exists": True,
#                 "reason": "可信政府或公司登記來源中出現完整相符公司名稱",
#                 "source": "tavily_trusted",
#                 "matched_url": hit["url"],
#                 "matched_title": hit["title"],
#                 "evidence": trusted_hits[:3],
#             }
#             _company_existence_cache[cache_key] = result
#             return result

#         # 2. 一般支援來源有完整公司名稱，但不是官方登記來源
#         if supporting_hits:
#             hit = supporting_hits[0]
#             result = {
#                 "exists": True,
#                 "reason": "搜尋結果有完整相符公司名稱，但來源不是政府或公司登記官方資料",
#                 "source": "tavily_supporting",
#                 "matched_url": hit["url"],
#                 "matched_title": hit["title"],
#                 "evidence": supporting_hits[:3],
#             }
#             _company_existence_cache[cache_key] = result
#             return result

#         # 3. 其他網站有完整公司名稱，但可信度不足
#         if other_hits:
#             hit = other_hits[0]
#             result = {
#                 "exists": None,
#                 "reason": "搜尋結果有完整相符公司名稱，但來源可信度不足",
#                 "source": "tavily_other",
#                 "matched_url": hit["url"],
#                 "matched_title": hit["title"],
#                 "evidence": other_hits[:3],
#             }
#             _company_existence_cache[cache_key] = result
#             return result

#         # 4. 完全沒有完整相符結果
#         result = {
#             "exists": False,
#             "reason": "Tavily 搜尋結果中沒有找到完整相符公司名稱",
#             "source": "tavily",
#             "matched_url": None,
#             "matched_title": None,
#             "evidence": [],
#         }
#         _company_existence_cache[cache_key] = result
#         return result

#     except Exception as e:
#         result = {
#             "exists": None,
#             "reason": f"exception: {e}",
#             "source": "tavily",
#         }
#         _company_existence_cache[cache_key] = result
#         return result
    
def verify_company_existence_by_model(company_name: str) -> dict:
    """
    用模型做「保守」存在性/可疑性判斷：
    - true  : 很有把握存在/合理（通常很少）
    - false : 明顯不像公司名/明顯錯
    - null  : 無法確定（不要猜）
    回傳: {"exists": True/False/None, "reason": "..."}
    """
    name = (company_name or "").strip()
    if not name:
        return {"exists": None, "reason": "empty"}

    key = _normalize_company_name_for_check(name)
    if key in _company_existence_cache:
        return _company_existence_cache[key]

    #請只根據公司名稱字面是否合理，做保守判斷。
    prompt = f"""
請判斷是否存在此公司名稱

規則：
- 若名稱包含明顯 OCR 雜訊/符號，或不像台灣公司名稱 → exists=false
- 若名稱看起來合理，但你無法確定真實存在 → exists=null（不要猜）
- 只有在你非常確定是明確存在且常見的公司時 → exists=true（通常很少）

公司名稱：{name}

請只回傳 JSON，不要加任何說明：
{{
  "exists": true 或 false 或 null,
  "reason": "一句話原因"
}}"""

    try:
        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL3,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0
        )

        content = (response.choices[0].message.content or "").strip()

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            result = {"exists": None, "reason": "non-json"}
            _company_existence_cache[key] = result
            return result

        raw = json.loads(json_match.group())
        exists = raw.get("exists")

        # 兼容模型可能回 "null"/"true"/"false" 字串
        if isinstance(exists, str):
            low = exists.strip().lower()
            if low == "true":
                exists = True
            elif low == "false":
                exists = False
            else:
                exists = None
        elif exists is True:
            exists = True
        elif exists is False:
            exists = False
        else:
            exists = None

        result = {"exists": exists, "reason": str(raw.get("reason", "")).strip()}
        _company_existence_cache[key] = result
        return result

    except Exception as e:
        result = {"exists": None, "reason": f"exception: {e}"}
        _company_existence_cache[key] = result
        return result


def extract_fields_from_image_region(
    image_pil: Image.Image,
    failed_fields: list
) -> dict:
    """
    單次 VLM 辨識，回傳擷取結果
    重試邏輯由 app.py 的比對結果決定
    """
    #     "發票日期": """發票上的開立日期
    #    - 可能是民國格式（如「115年03月17日」）或西元格式（如「2026/03/17」或「2026-03-17」）
    #    - 直接輸出原始格式，不需轉換
    #    - 找不到填 null
    #    - 請逐字確認格式正確，不要輸出到錯誤內容
    #    - 其中內容需包含年,月,日
    field_instructions = {

        "金額大寫中文": """
            - 金額大寫中文：只輸出繁體中文財務大寫金額本身，不要包含「新臺幣」前綴。

            - 此欄位所表示的數值，必須與「合計金額」完全相同。
            - 金額大寫中文是「合計金額」的中文財務大寫表示，
            不是未稅金額，也不是稅額。

            - 請先逐字辨識圖片上實際印出的金額大寫中文。
            - 如果圖片中的金額大寫中文清楚，而且與合計金額一致，請使用圖片上的文字。
            - 如果金額大寫中文模糊、缺字、形近字辨識錯誤，或與合計金額不一致，
            但合計金額清楚可信，請將合計金額轉換為繁體中文財務大寫後輸出。

            - 例如：
            合計金額 35910
            必須輸出：參萬伍仟玖佰壹拾元整

            - 禁止將未稅金額 34200 轉換成金額大寫中文。
            - 禁止混合未稅金額、稅額及合計金額的數字。
            - 若合計金額也無法可靠辨識，才輸出 null。

            - 使用以下繁體中文財務數字：
            零、壹、貳、參、肆、伍、陸、柒、捌、玖、拾、佰、仟、萬、億、元、整。

            - 表格框線、底線、刪除線、水平長線及欄位分隔線都不是文字，必須完全忽略。
            """,

        "未稅金額": """- 未稅金額：未含稅的銷售金額，輸出純數字並去除逗號。
        - 請先辨識圖片上實際印出的未稅金額。
        - 未稅金額、稅額、合計金額彼此具有數學關係，請搭配其他金額欄位交叉確認。
        - 如果圖片上的未稅金額模糊、疑似誤印或信心不足，但稅額清楚可信，且本張發票為一般 5% 應稅發票，可以使用：
        未稅金額 = 稅額 ÷ 0.05
        進行推算。
        - 例如稅額為 1710，則未稅金額可推算為 34200。
        - 如果合計金額也清楚，必須再確認：
        未稅金額 + 稅額 = 合計金額。
        - 若圖片直接辨識結果與推算結果衝突，請比較各欄位的清晰程度與可信度，採用整體數學關係最合理的結果。
        - 若無法確認為 5% 應稅，且其他金額也不足以交叉驗證，輸出 null。""",

        "稅額": """- 稅額：發票上的營業稅金額，輸出純數字並去除逗號。
        - 請優先逐字辨識圖片上實際印出的稅額。
        - 如果稅額在圖片中最清楚，應將稅額視為主要判斷依據，再交叉推算未稅金額與合計金額。
        - 一般 5% 應稅發票可使用：
        稅額 = 未稅金額 × 0.05
        - 也可使用：
        稅額 = 合計金額 - 未稅金額
        - 如果圖片上的稅額清楚可辨，不可因其他模糊欄位而任意修改稅額。
        - 如果稅額本身模糊，但未稅金額與合計金額清楚可信，可以使用兩者相減推算稅額。
        - 若多個可能值中有一個符合 5% 稅率，且同時符合未稅金額加稅額等於合計金額，應優先採用該合理值。""",

        "合計金額": """- 合計金額：含稅總計金額，輸出純數字並去除逗號。
        - 請先辨識圖片上實際印出的合計金額。
        - 合計金額應符合：
        合計金額 = 未稅金額 + 稅額。
        - 如果合計金額模糊、疑似誤印或信心不足，但未稅金額與稅額較清楚可信，可以使用兩者相加推算。
        - 如果只有稅額清楚可信，且確認為一般 5% 應稅發票，可以先推算：
        未稅金額 = 稅額 ÷ 0.05
        再推算：
        合計金額 = 未稅金額 + 稅額。
        - 例如稅額為 1710：
        未稅金額 = 1710 ÷ 0.05 = 34200
        合計金額 = 34200 + 1710 = 35910。
        - 如果圖片直接辨識的合計金額與推算結果衝突，請比較圖片清晰程度，採用整體稅率與加總關係最合理的結果。
        - 若無法確認稅率，且其他欄位不足以交叉驗證，輸出 null。""",

        
        #"未稅金額":    "- 未稅金額：未含稅的銷售金額（純數字，去除逗號）",
        #"稅額":       "- 稅額：營業稅金額（純數字，去除逗號）",
        #"合計金額":    "- 合計金額：含稅總計金額（純數字，去除逗號）",
        "年度期間":    "- 年度期間：格式為「民國年份年MM-MM月」，例如「115年03-04月」",
          "發票號碼":    "- 發票號碼：2個英文字母+8個數字，例如「AY83205584」",
        "買方統編":    """- 買方統編：買方的8位數字統一編號
                            - 注意：統一編號只會有八碼
                            - 注意：請逐個數字確認
                            - 注意：如果有找到「05637971」這串數字，這個就是買方統編""",
        "賣方統編":    """- 賣方統編：賣方的8位數字統一編號
        - 賣方統編號碼不會是「05637971」，「05637971」是買方統編，請勿誤判
        - 注意：部分發票賣方統編會以「#」開頭並緊接在日期/時間戳記之後，例如「2026-05-04 16:38#23762748」，此時請擷取「#」後面的8位數字（23762748），輸出時不要包含「#」符號
        - 注意：部分發票賣方統編會寫在「NO.」或「No.」之後，例如「TEL:03-3509780 NO.84673493」，此時請擷取「NO.」後面的8位數字（84673493），輸出時不要包含「NO.」前綴""",
        "買方公司名稱": """- 買方公司名稱：完整買方公司名稱
        - 請逐字辨識，不要依常見公司名稱自行猜測。

        - 特別注意以下容易混淆的形近字：
        「 燿／耀」
        - 遇到上述字形時，請放大觀察筆畫後再判斷。""",
        "賣方公司名稱": """- 賣方公司名稱：完整賣方公司名稱
        - 特別注意以下容易混淆的形近字：
        「 佳／全／住／佺」
        - 遇到上述字形時，請放大觀察筆畫後再判斷。""",
        #"賣方公司名稱": """- 賣方公司名稱：完整賣方公司名稱。
        #- 請逐字的確認，如果有出現類似「晁」「鼎」字，請確認到底是哪一個字""",
        "營業稅稅別判斷": """- 營業稅稅別判斷：請判斷發票上勾選的是「應稅」、「零稅率」或「免稅」。

   【非常重要：禁止推論】
  - 不可以根據「稅額是否大於 0」判斷為應稅。
  - 不可以根據「有營業稅金額」判斷為應稅。
  - 不可以根據「發票類型」、「金額欄位」、「稅法常識」、「一般商業邏輯」推論稅別。
  - 不可以因為看到「營業稅：xxx 元」就輸出「應稅」。
  - 不可以自行補判斷沒有明確顯示的稅別。
  - 只有在圖片中清楚看到稅別勾選框，且能確認哪一個稅別被勾選時，才可以輸出應稅、零稅率或免稅。
  - 如果圖片中沒有顯示稅別勾選框、稅別列被裁切、模糊、遮蔽、看不到勾選位置，或只能看到稅額但看不到勾選框，一律輸出 null。 

  這張台灣統一發票的營業稅稅別列，版型通常由左到右排列如下：

  營業稅 | 應稅 | 應稅的勾選框 | 零稅率 | 零稅率的勾選框 | 免稅 | 免稅的勾選框

  非常重要：
  - 勾選框是在該選項文字的右邊。
  - 勾選框屬於它左邊最近的稅別文字。
  - 不要因為勾選符號出現在某個文字左邊，就判斷成右邊那個文字。
  - 如果看到「應稅  √  零稅率」，代表勾選的是「應稅」，不是「零稅率」。
  - 如果看到「零稅率  √  免稅」，才代表勾選的是「零稅率」。
  - 如果看到「免稅  √」，才代表勾選的是「免稅」。

  判斷規則：
  1. √ / ✓ / 勾選符號在「應稅」右邊，且在「零稅率」左邊，輸出「應稅」。
  2. √ / ✓ / 勾選符號在「零稅率」右邊，且在「免稅」左邊，輸出「零稅率」。
  3. √ / ✓ / 勾選符號在「免稅」右邊，輸出「免稅」。
  4. 如果完全找不到勾選符號，輸出 null。

  只輸出：應稅 或 零稅率 或 免稅 或 null""",
        "明細項目":    """- 明細項目：每筆包含品名、數量、單價、金額，輸出為 JSON 陣列
  注意：
  - 對於容易混淆的中文字，請特別逐字確認，例如：「腦」不要誤判為「磁」、「單」不要誤判為「軍」、「劑」不要誤判為「齊」。
  - 品名請完整輸出，包含前面的料號數字
  - 品名如果換行請合併成完整品名
  - 數量請保留單位（如 40.0 LT）
  - 單價、金額為純數字（去除逗號）
  - 不要包含合計列、稅額列
  - 數量的表格內可能除了數字外還會有單位，只需辨識出數字即可
  - 數量、單價、金額三者間存在「數量 × 單價 = 金額」的關係，可用來檢查辨識結果，但不能取代逐字辨識：
    1. 請先各自逐字辨識數量、單價、金額三個數字，不要一開始就用計算代替辨識。
    2. 辨識完後，用「數量 × 單價 = 金額」檢查三者是否吻合。
    3. 「金額」一律以圖片上實際辨識到的文字為準，絕對不能用數量、單價反推或修改金額。
    4. 只有「數量」或「單價」其中一個模糊、筆畫不清、或形近字造成不確定時，
       才可以用「金額」與另一個清楚、有把握的數字（單價或數量）反推、修正那一個不確定的數字。
    5. 如果數量、單價都清楚可信、只是與金額對不上（例如有折扣、單位換算等情況），
       請照實輸出圖片上的文字，不要自行修改任何一個。
  例如：[{"品名": "414288 ENTEK劑", "數量": "40.0 LT", "單價": "1470", "金額": "58800"}]"""
    }

    json_template = {}
    for field in failed_fields:
        if field == "明細項目":
            json_template[field] = [{"品名": "<值>", "數量": "<原始文字>", "單價": "<純數字>", "金額": "<純數字>"}]
        elif field == "營業稅稅別判斷":
            json_template[field] = "<應稅 或 零稅率 或 免稅 或 null>"
        else:
            json_template[field] = "<值或null>"

    json_template["欄位信心值"] = {field: "<0~1或null>" for field in failed_fields}
    json_template["reason"] = "<簡短說明擷取依據或找不到的原因>"

    instructions = "\n".join([
        field_instructions[f] for f in failed_fields if f in field_instructions
    ])

    tax_layout_hint = ""

    if "營業稅稅別判斷" in failed_fields:
        tax_layout_hint = """
    【營業稅稅別判斷特別規則】

    請特別注意台灣發票的稅別勾選框位置。

    本版型不是「勾選框在文字左邊」。
    本版型是「文字在左，勾選框在右」。

    也就是：

    應稅 | 應稅勾選框 | 零稅率 | 零稅率勾選框 | 免稅 | 免稅勾選框

    因此：
    - 如果勾選符號位於「應稅」和「零稅率」之間，答案是「應稅」。
    - 如果勾選符號位於「零稅率」和「免稅」之間，答案是「零稅率」。
    - 如果勾選符號位於「免稅」右邊，答案是「免稅」。

    不要把「√ 零稅率」誤判成零稅率，因為該 √ 可能是左側「應稅」的勾選框。
    """
        
    amount_relation_hint = ""

    amount_fields = {"未稅金額", "稅額", "合計金額","金額大寫中文"}

    if amount_fields.intersection(failed_fields):
        amount_relation_hint = """
    【未稅金額、稅額、合計金額交叉判斷規則】

    未稅金額、稅額與合計金額彼此具有關聯，請不要將三個欄位完全獨立判斷。

    請依照以下順序處理：

    1. 先分別觀察圖片中的：
    - 未稅金額
    - 稅額
    - 合計金額

    2. 判斷哪一個欄位在圖片中最清楚、最可信，將該欄位作為主要依據。

    3. 金額應符合以下基本關係：

    合計金額 = 未稅金額 + 稅額

    4. 如果可以確認為一般 5% 應稅發票，亦可使用：

    稅額 = 未稅金額 × 0.05

    未稅金額 = 稅額 ÷ 0.05

    5. 如果稅額最清楚，而未稅金額與合計金額模糊或疑似有誤，可以依序推算：

    未稅金額 = 稅額 ÷ 0.05

    合計金額 = 未稅金額 + 稅額

    6. 例如圖片中清楚辨識到稅額為 1710：

    未稅金額 = 1710 ÷ 0.05 = 34200

    合計金額 = 34200 + 1710 = 35910

    因此可輸出：

    未稅金額：34200
    稅額：1710
    合計金額：35910

    6-1.【強制檢查，即使加總關係成立也一定要做】
    「合計金額 = 未稅金額 + 稅額」這個加總關係只能證明三個數字彼此「內部一致」，
    不能證明這三個數字本身是正確的（三個數字可能同時抄錯、位數錯位或多一位數，
    只要抄錯得剛好還是能加總起來）。

    因此，只要稅額有值（大於 0）且稅額本身辨識信心足夠高，
    無論未稅金額與合計金額的加總關係是否已經成立，都必須「另外」執行以下第二重檢查：

        預期未稅金額 = 稅額 ÷ 0.05

    將「預期未稅金額」與圖片直接辨識到的未稅金額比較（允許因四捨五入產生的 1 元以內誤差）：

    - 若兩者相符 → 加總關係與 5% 稅率關係都成立，可直接採用原辨識結果。
    - 若兩者不符 → 代表未稅金額（進而合計金額）辨識錯誤，即使加總關係表面成立，
      也必須改用「稅額 ÷ 0.05」推算出的未稅金額為準，並重新計算：

        合計金額 = 推算未稅金額 + 稅額

    錯誤示範（禁止輸出這種結果）：
    圖片辨識稅額 = 11750，但未稅金額被誤讀成 35000，合計金額被誤讀成 46750。
    雖然 35000 + 11750 = 46750（加總關係成立），但：
        11750 ÷ 0.05 = 235000 ≠ 35000
    兩者不符，代表 35000 與 46750 都是錯的，正確結果應為：
        未稅金額 = 11750 ÷ 0.05 = 235000
        合計金額 = 235000 + 11750 = 246750
    絕對不可以因為加總關係成立，就誤以為 35000 / 11750 / 46750 是正確答案。

    6-2.【欄位歸屬鐵則，不可配錯欄位】
    「稅額 ÷ 0.05」這個算式，算出來的數字定義上「一定是未稅金額」，绝對不是合計金額，
    也不是其他任何欄位。

    即使「稅額 ÷ 0.05」算出來的數字，剛好與圖片上直接讀到的「合計金額」數字相同，
    也絕對不可以把這個數字當作合計金額使用，更不可以反過來用
    「（圖片讀到的）合計金額 － 稅額」去推算未稅金額 ——
    這種做法等於是先假設圖片讀到的合計金額沒有錯，但圖片讀到的合計金額本身
    正是可能出錯的欄位，不能拿來當作推算的起點。

    正確作法永遠是：
        1. 未稅金額 = 稅額 ÷ 0.05（以稅額為唯一推算起點）
        2. 合計金額 = 未稅金額 + 稅額（用算出來的未稅金額往下加，不是用原本讀到的合計金額）

    錯誤示範（禁止輸出這種結果）：
    圖片辨識稅額 = 1530，圖片直接讀到的合計金額看起來是 30600。
    計算 1530 ÷ 0.05 = 30600 後，發現這個數字剛好等於圖片讀到的合計金額，
    於是誤以為「30600 就是合計金額」，再用 30600 － 1530 = 29070 當作未稅金額。
    這是錯的：1530 ÷ 0.05 算出來的 30600 定義上就是未稅金額本身，
    不是合計金額，正確結果應為：
        未稅金額 = 1530 ÷ 0.05 = 30600
        合計金額 = 30600 + 1530 = 32130
    絕對不可以因為算出來的數字與圖片讀到的合計金額相同，就把它誤配成合計金額欄位。

    7. 如果圖片直接辨識出的某個金額，與另外兩個欄位的數學關係不一致，請比較：

    - 哪個欄位字體最清楚
    - 哪個欄位信心值最高
    - 是否符合 5% 稅率（見第 6-1、6-2 點的強制檢查，優先權高於單純的加總關係）
    - 是否符合未稅金額加稅額等於合計金額

    8. 若其中一個直接辨識值疑似筆誤、模糊或多辨識一個數字，可以採用整體數學關係較合理的推算值；
       當加總關係與 5% 稅率關係互相衝突時，以 5% 稅率關係（第 6-1、6-2 點）為準。

    9. 不可在沒有任何可靠金額依據時自行猜測。

    10. 如果無法確認本張發票適用 5% 稅率，則不可只依 5% 稅率推算；此時只能使用：

        合計金額 = 未稅金額 + 稅額

    11. 欄位信心值請依來源設定：

        - 圖片清楚直接辨識：0.90～1.00
        - 根據兩個清楚金額交叉推算：0.75～0.89
        - 只根據稅額與 5% 稅率推算：0.65～0.80
        - 無法可靠判斷：輸出 null

    12. reason 必須清楚說明：
        - 哪個金額是圖片直接辨識
        - 哪個金額是利用公式推算
        - 使用了哪一條公式
        - 第 6-1、6-2 點的強制 5% 檢查結果（相符或不符，若不符是否已改用推算值）
     13. 金額大寫中文必須與合計金額表示完全相同的數值。

        金額大寫中文 = 合計金額轉換成繁體中文財務大寫

    14. 金額大寫中文只能參考「合計金額」進行轉換，
        不可以參考未稅金額或稅額直接產生。

    15. 例如：

        未稅金額 = 34200
        稅額 = 1710
        合計金額 = 35910

        則：

        金額大寫中文 = 參萬伍仟玖佰壹拾元整

        不可以輸出：
        參萬肆仟貳佰元整

        因為參萬肆仟貳佰元整代表的是未稅金額，不是合計金額。

    16. 如果圖片直接辨識出的金額大寫中文與合計金額不一致：

        - 合計金額清楚且可信時，使用合計金額轉換後的中文財務大寫。
        - 合計金額不清楚時，不可任意推算，輸出 null。

    17. reason 必須說明金額大寫中文是：

        - 圖片直接辨識且與合計金額一致；或
        - 根據合計金額轉換產生。
    """



    prompt = f"""這是一張台灣統一發票圖片，請仔細閱讀圖片內容，擷取以下欄位。

    

    {tax_layout_hint}

    {amount_relation_hint}

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

        result["__field_confidence__"] = _normalize_field_confidence_map(
            result.get("欄位信心值"),
            failed_fields
        )
        result.pop("欄位信心值", None)
        # reason 保留在 result 中供呼叫端使用
        if "reason" not in result:
            result["reason"] = None

        return result

    except json.JSONDecodeError as e:
        print(f"⚠️  [VLM保底] JSON 解析失敗：{e}")
        return {}
    except Exception as e:
        print(f"⚠️  [VLM保底] 呼叫失敗：{e}")
        return {}


def extract_invoice_number_from_image(image_pil: Image.Image, known_prefix: str = None) -> dict:
    """專用 VLM：只辨識發票號碼，避免通用欄位提示分散注意力。"""
    prefix_hint = ""
    if known_prefix:
        prefix_hint = f"\n已知字軌英文字母可能是「{known_prefix}」。如果你只看到 8 位數字，請和 {known_prefix} 合併成完整發票號碼。"

    prompt = f"""請只判斷這張圖片中的發票號碼。

發票號碼格式：
- 2 個英文字母 + 8 個數字
- 例如 AY83205584、ZX11414651

請特別看圖片左上方或上方偏左的發票字軌區。
如果看到英文字母（例如 ZX），並且右側或附近斜線底紋區有 8 位數字，請合併成完整發票號碼。{prefix_hint}

不要輸出：
- 買方統編
- 賣方統編
- 日期
- 電話
- 金額
- 地址

請只回傳 JSON，不要加任何說明：
{{
  "發票號碼": "<2個英文字母+8個數字或null>",
    "reason": "<簡短原因>",
    "信心值": "<0~1或null>"
}}"""

    try:
        b64_image = image_to_base64(image_pil)

        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL2,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }],
            max_tokens=256,
            temperature=0.0
        )

        content = (response.choices[0].message.content or "").strip()
        print(f"[VLM發票號碼專用] 回應:\n{content}\n")
        

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            print("⚠️  [VLM發票號碼專用] 找不到 JSON")
            return {"發票號碼": None, "reason": "VLM回應非JSON"}

        result = json.loads(json_match.group())
        return {
            "發票號碼": result.get("發票號碼"),
            "reason": result.get("reason", ""),
            "__field_confidence__": {"發票號碼": _to_confidence_01(result.get("信心值"))} if _to_confidence_01(result.get("信心值")) is not None else {}
        }

    except Exception as e:
        print(f"⚠️  [VLM發票號碼專用] 呼叫失敗：{e}")
        return {"發票號碼": None, "reason": f"VLM呼叫失敗：{e}"}


def extract_tax_type_from_image(image_pil: Image.Image) -> dict:
    """專用 VLM：對裁切後的稅別區塊圖片做營業稅稅別判斷。"""
    prompt = """請判斷這張圖片中的營業稅稅別勾選結果。

    
  【非常重要：禁止推論】
  - 不可以根據「稅額是否大於 0」判斷為應稅。
  - 不可以根據「有營業稅金額」判斷為應稅。
  - 不可以根據「發票類型」、「金額欄位」、「稅法常識」、「一般商業邏輯」推論稅別。
  - 不可以因為看到「營業稅：xxx 元」就輸出「應稅」。
  - 不可以自行補判斷沒有明確顯示的稅別。
  - 只有在圖片中清楚看到稅別勾選框，且能確認哪一個稅別被勾選時，才可以輸出應稅、零稅率或免稅。
  - 如果圖片中沒有顯示稅別勾選框、稅別列被裁切、模糊、遮蔽、看不到勾選位置，或只能看到稅額但看不到勾選框，一律輸出 null。 
    

重要：這張圖片是裁切自台灣統一發票的稅別列區塊。

布局由左至右排列如下：
  營業稅 | 應稅 | 應稅勾選框 | 零稅率 | 零稅率勾選框 | 免稅 | 免稅勾選框

視覺判斷規則：
- 勾選框對應它左邊的選項文字（不是右邊）
- 勾選符號（√ / ✓ / X 或填入的內容）在「應稅」文字右還且在「零稅率」文字左還 → 勾選的是「應稅」
- 勾選符號在「零稅率」右還且在「免稅」左還 → 勾選的是「零稅率」
- 勾選符號在「免稅」右還 → 勾選的是「免稅」
- 完全找不到勾選符號 → 輸出 null

請只回傳 JSON，不要加任何說明：
{
  "營業稅稅別判斷": "應稅 或 零稅率 或 免稅 或 null",
  "reason": "簡短判斷原因",
  "信心値": "<0~1>"
}"""

    try:
        b64_image = image_to_base64(image_pil)

        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL2,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }],
            max_tokens=256,
            temperature=0.0
        )

        content = (response.choices[0].message.content or "").strip()
        print(f"[VLM稅別專用] 回應:\n{content}\n")

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            print("⚠️  [VLM稅別專用] 找不到 JSON")
            return {"營業稅稅別判斷": None, "reason": "VLM回應非JSON"}

        result = json.loads(json_match.group())
        return {
            "營業稅稅別判斷": result.get("營業稅稅別判斷"),
            "reason": result.get("reason", ""),
            "__field_confidence__": {"營業稅稅別判斷": _to_confidence_01(result.get("信心値"))} if _to_confidence_01(result.get("信心値")) is not None else {}
        }

    except Exception as e:
        print(f"⚠️  [VLM稅別專用] 呼叫失敗：{e}")
        return {"營業稅稅別判斷": None, "reason": f"VLM呼叫失敗：{e}"}


def detect_total_ntd_text(image_pil: Image.Image) -> dict:
    """用 VLM 判斷發票是否出現「總計新臺幣/總計新台幣」字樣。"""
    prompt = """請判斷這張台灣統一發票圖片中，是否有出現「總計新臺幣」或「總計新台幣」字樣。

請只回傳 JSON，不要加任何說明：
{
  "has_total_ntd_text": true 或 false,
  "matched_text": "看到的文字或null",
  "reason": "簡短原因"
}"""

    try:
        b64_image = image_to_base64(image_pil)
        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL2,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }],
            max_tokens=256,
            temperature=0.0
        )

        content = (response.choices[0].message.content or "").strip()
        print(f"[VLM總計新臺幣判斷] 回應:\n{content}\n")

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            return {"has_total_ntd_text": False, "matched_text": None, "reason": "VLM回應非JSON"}

        result = json.loads(json_match.group())
        return {
            "has_total_ntd_text": bool(result.get("has_total_ntd_text", False)),
            "matched_text": result.get("matched_text"),
            "reason": result.get("reason", "")
        }

    except Exception as e:
        print(f"⚠️  [VLM總計新臺幣判斷] 呼叫失敗：{e}")
        return {"has_total_ntd_text": False, "matched_text": None, "reason": f"VLM呼叫失敗：{e}"}

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
            "confidence":            float,  # 0.0 ~ 1.0
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
  "confidence": <0.0 到 1.0 之間的數字，代表判斷信心度>,
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

        # 確保 confidence 是 float
        if "confidence" in result:
            c = result["confidence"]
            if isinstance(c, str):
                result["confidence"] = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(c.lower(), 0.5)
            else:
                result["confidence"] = float(c)
        result["raw_response"] = raw_text
        return result

    except json.JSONDecodeError:
        # 解析失敗時，嘗試從文字判斷
        has_multiple = any(kw in raw_text for kw in ["多張", "兩張", "2張", "multiple", "more than one"])
        return {
            "has_multiple_invoices": has_multiple,
            "invoice_count":         -1,       # 無法確定
            "confidence":            0.3,
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