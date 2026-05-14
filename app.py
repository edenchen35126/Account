# =========================
# 套件引入
# =========================
from paddleocr import PaddleOCR          # OCR 辨識引擎
from pdf2image import convert_from_path  # PDF 轉圖片
from opencc import OpenCC                # 簡體轉繁體
import cv2                               # 圖像處理
import numpy as np                       # 數值運算
import os                                # 檔案系統操作
import json                              # JSON 讀寫
import re                                # 正則表達式
from PIL import Image, ImageDraw, ImageFont  # 圖片繪製與字型
import pandas as pd                      # Excel 讀取

# =========================
# 建立資料夾
# =========================
os.makedirs("jpg_pages", exist_ok=True)   # PDF 轉出的圖片存放處
os.makedirs("output", exist_ok=True)      # 視覺化結果存放處
os.makedirs("json_output", exist_ok=True) # JSON 結果存放處

# =========================
# OCR 初始化
# =========================
ocr = PaddleOCR(
    use_doc_orientation_classify=False,  # 不做文件方向分類
    use_doc_unwarping=False,             # 不做文件展平
    use_textline_orientation=False       # 不做文字行方向判斷
)

# =========================
# 簡體 → 繁體轉換器
# =========================
cc = OpenCC('s2t')  # s2t = Simplified to Traditional

# =========================
# 目標 PDF 設定
# =========================
pdf_path = "file/split_output/scan-00003/page_1.pdf"  # 要處理的 PDF 路徑
pdf_name = "page_1"                                    # 輸出檔名前綴

# =========================
# PDF → 圖片
# =========================
pages = convert_from_path(
    pdf_path,
    dpi=700,  # 解析度，越高品質越好但速度越慢，建議 300
    poppler_path="Release-25.12.0-0/poppler-25.12.0/Library/bin"  # Poppler 執行檔路徑
)



# # =========================
# # 目標檔案設定
# # =========================
# input_path = "file/invoices/page1_invoice3.png"  # 可以是 PDF 或圖片
# input_name = os.path.splitext(os.path.basename(input_path))[0]  # 自動取檔名

# # =========================
# # PDF → 圖片 / 直接讀圖片
# # =========================
# IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ext = os.path.splitext(input_path)[1].lower()

# if ext == ".pdf":
#     pages = convert_from_path(
#         input_path,
#         dpi=700,
#         poppler_path="Release-25.12.0-0/poppler-25.12.0/Library/bin"
#     )
#     print(f"PDF 模式：共 {len(pages)} 頁")

# elif ext in IMAGE_EXTENSIONS:
#     img_pil = Image.open(input_path).convert("RGB")
#     pages = [img_pil]  # 包成 list，讓後續迴圈邏輯不變
#     print(f"圖片模式：{input_path}")

# else:
#     raise ValueError(f"不支援的檔案格式：{ext}")


# =========================
# 讀取 Excel 標準答案
# =========================
def load_excel_standard(excel_path):
    """
    讀取標準答案 Excel，以發票號碼為 key 建立查詢字典
    
    Args:
        excel_path: Excel 檔案路徑
    Returns:
        dict: { 發票號碼: { 欄位: 值, ... } }
    """
    df = pd.read_excel(excel_path, dtype=str)
    df.columns = df.columns.str.strip()  # 去除欄位名稱前後空白

    standard_dict = {}
    for _, row in df.iterrows():
        invoice_no = str(row.get("發票號碼", "")).strip()
        if not invoice_no:
            continue  # 跳過空白列
        standard_dict[invoice_no] = {
            "發票號碼":    invoice_no,
            "金額大寫中文": str(row.get("金額大寫中文", "")).strip(),
            "年度期間":    str(row.get("年度期間", "")).strip(),
            "廠商統編":    str(row.get("廠商統編", "")).strip(),   # 即賣方統編
            "廠商名稱":    str(row.get("廠商名稱", "")).strip(),
            "未稅金額":    str(row.get("未稅金額", "")).strip(),
            "稅額":       str(row.get("稅額", "")).strip(),
            "合計金額":    str(row.get("合計金額", "")).strip(),
        }

    print(f"已載入標準答案，共 {len(standard_dict)} 筆")
    return standard_dict

# 買方統編固定值（本系統的買方永遠是此統編）
BUYER_TAX_ID_FIXED = "05637971"
BUYER_COMPANY_NAME_FIXED = "燿華電子股份有限公司"
# =========================
# 與 Excel 標準答案比對
# =========================
def compare_with_standard(buyer_tax_id, seller_tax_id, buyer_company_name, seller_company_name, amount_validation, extracted_invoice_no, standard):
    """
    將 OCR 擷取結果與 Excel 標準答案逐欄比對
    
    Args:
        buyer_tax_id:        OCR 辨識到的買方統編
        seller_tax_id:       OCR 辨識到的賣方統編
        amount_validation:   金額檢核結果 dict
        extracted_invoice_no: OCR 辨識到的發票號碼
        standard:            Excel 標準答案 dict（單筆）
    Returns:
        dict: 各欄位比對結果
    """
    # 找不到對應標準答案時，直接回傳錯誤訊息
    if standard is None:
        return {"比對結果": f"找不到發票號碼 [{extracted_invoice_no}] 對應的標準答案"}

    compare = {}

    # --- 0. 發票號碼比對 ---
    std_invoice_no = standard.get("發票號碼", "").strip()
    compare["發票號碼"] = {
        "標準答案": std_invoice_no,
        "OCR結果":  extracted_invoice_no or "",
        "是否一致": (extracted_invoice_no == std_invoice_no)
    }

    # --- 1. 買方統編：OCR 結果與固定值 05637971 比對 ---
    compare["買方統編"] = {
        "標準答案": BUYER_TAX_ID_FIXED,
        "OCR結果":  buyer_tax_id or "",
        "是否一致": (buyer_tax_id == BUYER_TAX_ID_FIXED)
    }

    # --- 2. 買方公司名稱：OCR 結果與固定值比對 ---
    is_match, match_method = is_company_name_match(buyer_company_name, BUYER_COMPANY_NAME_FIXED)
    compare["買方公司名稱"] = {
        "標準答案": BUYER_COMPANY_NAME_FIXED,
        "OCR結果":  buyer_company_name or "",
        "比對方式": match_method,
        "是否一致": is_match
    }
    # compare["買方公司名稱"] = {
    #     "標準答案": BUYER_COMPANY_NAME_FIXED,
    #     "OCR結果":  buyer_company_name or "",
    #     "是否一致": (buyer_company_name == BUYER_COMPANY_NAME_FIXED)
    # }

    # --- 3. 賣方統編：OCR 結果與 Excel 廠商統編比對 ---
    std_seller_tax_id = standard.get("廠商統編", "").strip()
    compare["賣方統編"] = {
        "標準答案": std_seller_tax_id,
        "OCR結果":  seller_tax_id or "",
        "是否一致": (seller_tax_id == std_seller_tax_id)
    }

    # --- 4. 賣方公司名稱比對 ---
    std_seller_name = standard.get("廠商名稱", "").strip()
    is_seller_match, seller_match_method = is_company_name_match(seller_company_name, std_seller_name)
    compare["賣方公司名稱"] = {
        "標準答案": std_seller_name,
        "OCR結果":  seller_company_name or "",
        "比對方式": seller_match_method,
        "是否一致": is_seller_match
    }

    # # --- 5. 未稅金額比對 ---
    # std_sales = standard.get("未稅金額", "").replace(",", "").strip()
    # ocr_sales = str(amount_validation.get("sales_amount") or "")
    # compare["未稅金額"] = {
    #     "標準答案": std_sales,
    #     "OCR結果":  ocr_sales,
    #     "是否一致": ocr_sales == std_sales
    # }

    # # --- 6. 稅額比對 ---
    # std_tax = standard.get("稅額", "").replace(",", "").strip()
    # ocr_tax = str(amount_validation.get("tax_amount") or "")
    # compare["稅額"] = {
    #     "標準答案": std_tax,
    #     "OCR結果":  ocr_tax,
    #     "是否一致": ocr_tax == std_tax
    # }

    # # --- 7. 合計金額比對 ---
    # std_total = standard.get("合計金額", "").replace(",", "").strip()
    # ocr_total = str(amount_validation.get("total_amount") or "")
    # compare["合計金額"] = {
    #     "標準答案": std_total,
    #     "OCR結果":  ocr_total,
    #     "是否一致": ocr_total == std_total
    # }

    # --- 8. 整體通過判斷：所有欄位都一致才算通過 ---
    compare["全部比對通過"] = all(
        v["是否一致"] for v in compare.values() if isinstance(v, dict)
    )

    return compare

# =========================
# OCR 擷取買方公司名稱的函式
# =========================
def extract_buyer_company_name(page_text_clean):
    lines = page_text_clean.split("\n")
    normalized_lines = [re.sub(r"\s+", "", l) for l in lines]

    for idx, line in enumerate(normalized_lines):

        is_buyer_line  = False
        content_idx    = idx

        # 正常情況：「買方」在同一行
        if re.search(r'買方|購買人', line):
            is_buyer_line = True

        # ✅ 跨行情況：「方:」單獨成行
        # 往前 5 行找是否有含「買」字的行（處理「統買地\n編:\n方:」的情況）
        elif re.search(r'^方[:：]', line):
            context_before = normalized_lines[max(0, idx-5):idx]
            if any('買' in l for l in context_before):
                is_buyer_line = True

        if not is_buyer_line:
            continue

        # 同行找公司名稱
        m = re.search(r'([\u4e00-\u9fff]{2,}(?:股份)?有限公司[\u4e00-\u9fff]*)', line)
        if m:
            return m.group(1)

        # 往後 3 行找
        for next_line in normalized_lines[idx+1:idx+4]:
            m = re.search(r'([\u4e00-\u9fff]{2,}(?:股份)?有限公司[\u4e00-\u9fff]*)', next_line)
            if m:
                return m.group(1)

    # Fallback：直接掃全文（處理 OCR 首字誤判，如燿→耀）
    full_text = "".join(normalized_lines)
    # 找所有出現的「XX電子股份有限公司」類型
    m = re.search(r'[\u4e00-\u9fff]{1,2}華電子股份有限公司', full_text)
    if m:
        return m.group()

    return None

# =========================
# 抓買方/賣方統一編號（上下文判斷）
# =========================
def extract_tax_id_by_context(page_text_clean):
    buyer_tax_id  = None
    seller_tax_id = None

    lines = page_text_clean.split("\n")
    normalized_lines = [re.sub(r"\s+", "", l) for l in lines]
    full_text = "".join(normalized_lines)

    # ✅ 買方統編：固定已知，直接比對全文
    if BUYER_TAX_ID_FIXED in full_text:
        buyer_tax_id = BUYER_TAX_ID_FIXED

    # ✅ 賣方統編：策略1 - 「統一編號:XXXXXXXX」格式（同行）
    for line in normalized_lines:
        m = re.search(r'統一?編號[:：](\d{8})', line)
        if m:
            candidate = m.group(1)
            if candidate != BUYER_TAX_ID_FIXED:
                seller_tax_id = candidate
                break

    # ✅ 賣方統編：策略2 - 跨行「統一編號\n:12345678」
    if seller_tax_id is None:
        for idx, line in enumerate(normalized_lines):
            if not re.search(r'統一?編號', line):
                continue
            for next_line in normalized_lines[idx+1:idx+4]:
                m = re.search(r'^[:：]?(\d{8})$', next_line)
                if m:
                    candidate = m.group(1)
                    if candidate != BUYER_TAX_ID_FIXED:
                        seller_tax_id = candidate
                    break
            if seller_tax_id:
                break

    # ✅ 賣方統編：策略3 - 「買方 賣方」同行，下一行緊接兩個統編
    # 處理「買方  賣方\n05637971  04406559」的格式
    if seller_tax_id is None:
        for idx, line in enumerate(normalized_lines):
            if not (re.search(r'買方', line) and re.search(r'賣方', line)):
                continue
            # 往後 3 行找含兩個 8 位數字的行
            for next_line in normalized_lines[idx+1:idx+4]:
                tax_ids = re.findall(r'\d{8}', next_line)
                if len(tax_ids) >= 2:
                    for tid in tax_ids:
                        if tid != BUYER_TAX_ID_FIXED:
                            seller_tax_id = tid
                            break
                elif len(tax_ids) == 1 and tax_ids[0] != BUYER_TAX_ID_FIXED:
                    # 只有一個且非買方統編
                    seller_tax_id = tax_ids[0]
                if seller_tax_id:
                    break
            if seller_tax_id:
                break

    # ✅ 賣方統編：策略4 - 「賣方」單獨行，往後幾行找 8 位數字
    # 處理「賣方\n04406559」的格式
    if seller_tax_id is None:
        for idx, line in enumerate(normalized_lines):
            if not re.search(r'^賣方$|^賣\s*方$', line):
                continue
            for next_line in normalized_lines[idx+1:idx+5]:
                m = re.search(r'(\d{8})', next_line)
                if m:
                    candidate = m.group(1)
                    if candidate != BUYER_TAX_ID_FIXED:
                        seller_tax_id = candidate
                        break
            if seller_tax_id:
                break

    # ✅ 賣方統編：策略5 - 「賣方XXXXXXXX」或「賣方:XXXXXXXX」直接在同行
    if seller_tax_id is None:
        for line in normalized_lines:
            m = re.search(r'賣方[:：]?(\d{8})', line)  # ✅ 冒號設為可選
            if m:
                candidate = m.group(1)
                if candidate != BUYER_TAX_ID_FIXED:
                    seller_tax_id = candidate
                    print(f"[DEBUG] 策略5 找到賣方統編: {seller_tax_id}")
                    break

    print(f"[DEBUG] 買方統編={buyer_tax_id}, 賣方統編={seller_tax_id}")
    return buyer_tax_id, seller_tax_id

# 買方公司名稱比對（忽略第一個字，因為 OCR 常誤判罕見字）
def is_company_name_match(ocr_name, standard_name):
    if not ocr_name or not standard_name:
        return False, "資料缺失"

    ocr = re.sub(r"\s+", "", ocr_name)
    std = re.sub(r"\s+", "", standard_name)

    # 1. 完全相等
    if ocr == std:
        return True, "完全相符"

    # 2. 忽略第一個字比對
    if len(ocr) >= 2 and len(std) >= 2:
        if ocr[1:] == std[1:]:
            return True, f"忽略首字相符（OCR首字:{ocr[0]} 標準首字:{std[0]}）"

    # 3. 標準答案包含於 OCR 結果中（OCR多抓了前後雜訊）
    if len(std) >= 4 and std in ocr:
        return True, "標準答案包含於OCR結果中"

    # 4. 忽略首字後，標準答案包含於 OCR 結果中
    if len(std) >= 4 and std[1:] in ocr:
        return True, f"忽略首字後包含於OCR結果中（標準首字:{std[0]}）"
    

    return False, "不相符"


def extract_seller_company_name(page_text_clean):
    VALID_EXTENSION_PATTERN = r'[\u4e00-\u9fff]{1,8}(分公司|辦事處|營業所|工廠|廠|事業部|物流中心|倉儲中心|貨櫃集散站|集散站|貨櫃場|營業處|服務中心|配送中心)'
    TAX_CATEGORY_PATTERN = r'^(應稅|免稅|零稅率|營業稅|銷售額合計|合計|總計|銷售額)$'
    MAX_FRAGMENT_LEN = 12

    lines = page_text_clean.split("\n")
    normalized_lines = [re.sub(r"\s+", "", l) for l in lines]

    print("=== [DEBUG] extract_seller_company_name ===")

    for idx, line in enumerate(normalized_lines):
        is_seller_line = False
        seller_content_idx = idx

        if (re.search(r'賣方|賣\s*方', line) or
            ('賣' in line and '方' in line and re.search(r'有限公司', line))):
            is_seller_line = True
        elif line == '賣':
            next_line = normalized_lines[idx+1] if idx+1 < len(normalized_lines) else ''
            if next_line.startswith('方'):
                is_seller_line = True
                seller_content_idx = idx + 1
        elif re.search(r'^營業人[:：]?', line):
            is_seller_line = True

        print(f"  [{idx}] normalized='{line}' | is_seller={is_seller_line}")

        if not is_seller_line:
            continue

        company_name = ""
        base_idx = seller_content_idx
        trigger_line = normalized_lines[seller_content_idx]

        m = re.search(r'([\u4e00-\u9fff]{2,}(?:股份)?有限公司[\u4e00-\u9fff]*)', trigger_line)
        if m:
            company_name = m.group(1)
            print(f"  → 同行找到: '{company_name}'")
        else:
            accumulated = re.sub(r'^[賣方:：\s]+', '', re.sub(r'[^\u4e00-\u9fff]', '', trigger_line))
            print(f"  → 累積起點: '{accumulated}'，往後掃描...")

            for offset in range(1, 10):
                next_idx = seller_content_idx + offset
                if next_idx >= len(normalized_lines):
                    break
                next_norm = normalized_lines[next_idx]

                if re.search(r'\d{4,}', next_norm):
                    continue
                if re.search(TAX_CATEGORY_PATTERN, next_norm):
                    print(f"    [{next_idx}] 跳過稅率分類詞: '{next_norm}'")
                    continue

                chinese_part = re.sub(r'[^\u4e00-\u9fff]', '', next_norm)

                if len(chinese_part) > MAX_FRAGMENT_LEN:
                    print(f"    [{next_idx}] 片段過長({len(chinese_part)}字)，停止: '{next_norm}'")
                    break

                if chinese_part:
                    accumulated += chinese_part
                    print(f"    [{next_idx}] 累積: '{accumulated}'")

                m = re.search(r'([\u4e00-\u9fff]{2,}(?:股份)?有限公司[\u4e00-\u9fff]*)', accumulated)
                if m:
                    company_name = m.group(1)
                    base_idx = next_idx
                    print(f"  → 累積後找到: '{company_name}'")
                    break

        if not company_name:
            print(f"  → 找不到公司名稱，continue")
            continue

        for next_line in normalized_lines[base_idx+1:base_idx+6]:
            if not next_line:
                continue
            if re.search(VALID_EXTENSION_PATTERN, next_line):
                company_name += next_line
                continue
            if re.search(r'\d', next_line):
                break
            if len(re.findall(r'[A-Za-z]', next_line)) >= 2:
                break
            continue

        print(f"  → 最終結果: '{company_name}'")
        print("===========================================")
        return company_name

    # ✅ Fallback：正常流程找不到時，掃全文找所有「有限公司」
    # 排除買方公司名稱，剩下的就是賣方
    print("  → Fallback：掃全文找賣方公司名稱")
    buyer_normalized = re.sub(r"\s+", "", BUYER_COMPANY_NAME_FIXED)
    for line in normalized_lines:
        m = re.search(r'([\u4e00-\u9fff]{2,}(?:股份)?有限公司[\u4e00-\u9fff]*)', line)
        if m:
            candidate = m.group(1)
            # 排除買方（允許首字誤判，用忽略首字比對排除）
            is_buyer = (
                candidate == buyer_normalized or
                (len(candidate) >= 2 and len(buyer_normalized) >= 2 and
                 candidate[1:] == buyer_normalized[1:])
            )
            if not is_buyer:
                print(f"  → Fallback 找到: '{candidate}'")
                print("===========================================")
                return candidate

    print("  → 未找到賣方行，回傳 None")
    print("===========================================")
    return None

# =========================
# 中文字型（視覺化用）
# =========================
font_path = r"C:\Windows\Fonts\msjh.ttc"
font       = ImageFont.truetype(font_path, 28)   # 大字（摘要資訊）
small_font = ImageFont.truetype(font_path, 20)   # 小字（OCR 標注）

# =========================
# 工具函式
# =========================
def clean_text(text: str) -> str:
    """將全形標點符號統一轉成半形，方便後續 Regex 比對"""
    text = text.replace("：", ":")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("，", ",")
    text = text.replace("。", ".")
    return text

def normalize_text(text: str) -> str:
    """clean_text 後再移除所有空白"""
    text = clean_text(text)
    text = re.sub(r"\s+", "", text)
    return text

def poly_to_bbox(poly):
    """將多邊形頂點座標轉換為 [x1, y1, x2, y2] 的外接矩形"""
    arr = np.array(poly)
    x1 = int(np.min(arr[:, 0]))
    y1 = int(np.min(arr[:, 1]))
    x2 = int(np.max(arr[:, 0]))
    y2 = int(np.max(arr[:, 1]))
    return [x1, y1, x2, y2]

def point_in_bbox(px, py, bbox):
    """判斷點 (px, py) 是否在 bbox 內"""
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2

def bbox_area(bbox):
    """計算 bbox 面積"""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)

def bbox_intersection_area(b1, b2):
    """計算兩個 bbox 的交集面積"""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0  # 無交集
    return (x2 - x1) * (y2 - y1)

def overlap_ratio(inner_bbox, outer_bbox):
    """
    計算 inner_bbox 有多少比例落在 outer_bbox 內
    回傳值介於 0.0 ~ 1.0
    """
    inter = bbox_intersection_area(inner_bbox, outer_bbox)
    area = bbox_area(inner_bbox)
    if area == 0:
        return 0.0
    return inter / area

def sort_bbox_texts(items, y_tolerance=10):
    """
    將 OCR items 依照閱讀順序排序（先由上到下分行，同行內由左到右）
    
    Args:
        items:       OCR item 清單，每個 item 需有 bbox 欄位
        y_tolerance: 同一行的 Y 座標允許誤差（像素）
    Returns:
        排序後的 item 清單
    """
    if not items:
        return []

    items = sorted(items, key=lambda x: x["bbox"][1])  # 先依 Y 排序
    lines = []

    for item in items:
        y1 = item["bbox"][1]
        placed = False

        # 找是否有已存在的行可以歸入（Y 差距在容忍範圍內）
        for line in lines:
            line_y = int(np.mean([t["bbox"][1] for t in line]))
            if abs(y1 - line_y) <= y_tolerance:
                line.append(item)
                placed = True
                break

        if not placed:
            lines.append([item])  # 建立新行

    # 每行內依 X 座標由左到右排序
    for line in lines:
        line.sort(key=lambda x: x["bbox"][0])

    # 行與行之間依最上方 Y 座標排序
    lines.sort(key=lambda line: min(t["bbox"][1] for t in line))

    result = []
    for line in lines:
        result.extend(line)

    return result

# =========================
# OCR 執行與解析
# =========================
def run_ocr(image_path):
    """
    對指定圖片執行 PaddleOCR，回傳結構化結果
    
    Returns:
        list of dict: [{ "text": str, "poly": list, "bbox": [x1,y1,x2,y2] }, ...]
    """
    result = ocr.predict(image_path)

    ocr_items = []

    for res in result:
        data = res.json
        if isinstance(data, str):
            data = json.loads(data)

        ocr_data = data.get("res", data)
        texts = ocr_data.get("rec_texts", [])
        polys = ocr_data.get("rec_polys", ocr_data.get("dt_polys", []))

        for text, poly in zip(texts, polys):
            bbox = poly_to_bbox(poly)
            ocr_items.append({
                "text": cc.convert(text),  # 簡體轉繁體
                "poly": np.array(poly).astype(int).tolist(),
                "bbox": bbox
            })

    return ocr_items

def extract_value_by_keyword(text: str, keyword_pattern: str, value_pattern: str):
    """
    在文字中找「關鍵字 + 值」的組合，回傳值的部分
    例如：「統一編號:12345678」→ 回傳「12345678」
    """
    pattern = rf'{keyword_pattern}\s*[:：]?\s*({value_pattern})'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else None

def extract_multiline_digits_after_keyword(text: str, keyword_pattern: str, digit_count: int = 8):
    """
    關鍵字後方跨行找指定位數的數字
    （因 OCR 有時候會把關鍵字和數字拆到不同行）
    """
    pattern = rf'{keyword_pattern}[\s:：\n]*([\d\s\n]{{{digit_count},{digit_count * 4}}})'
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None

    raw_value = match.group(1)
    digits_only = re.sub(r'\D', '', raw_value)  # 只保留數字

    if len(digits_only) >= digit_count:
        return digits_only[:digit_count]  # 取前 N 碼

    return None

def extract_fields_from_ocr_text(page_text_clean):
    field_rules = {
        "統一編號": {
            "keyword_pattern": r'統一\s*編號',
            "value_pattern": r'\d{8}'
        },
        "發票號碼": {
            "keyword_pattern": r'發票號碼|發票號',
            "value_pattern": r'[A-Z]{2}[-\s]?\d{8}'  # ✅ 允許中間有「-」或空白
        }
    }

    extracted_data = {}

    for field_name, rule in field_rules.items():
        value = extract_value_by_keyword(
            page_text_clean,
            rule["keyword_pattern"],
            rule["value_pattern"]
        )
        extracted_data[field_name] = value

    if extracted_data["統一編號"] is None:
        extracted_data["統一編號"] = extract_multiline_digits_after_keyword(
            page_text_clean, r'統一\s*編號', 8
        )

    # ✅ Fallback：掃全文找發票號碼，允許中間有「-」，找到後清除「-」
    if extracted_data["發票號碼"] is None:
        m = re.search(r'[A-Z]{2}-?\d{6,8}', page_text_clean)
        if m:
            raw = m.group()
            cleaned = raw.replace("-", "").replace(" ", "")
            extracted_data["發票號碼"] = cleaned
            print(f"[DEBUG] 發票號碼原始辨識: '{raw}' → 清除後: '{cleaned}'")

    return extracted_data

# =========================
# TSR（表格結構辨識）
# =========================
def merge_nearby_lines(lines, axis="x", gap_threshold=15):
    """
    將相近的線條合併為一條，避免重複偵測
    
    Args:
        lines:         線條清單（bbox 格式）
        axis:          合併方向，"x" 為垂直線，"y" 為水平線
        gap_threshold: 距離小於此值視為同一條線（像素）
    Returns:
        合併後的座標清單
    """
    if not lines:
        return []

    # 取線條中心座標
    if axis == "x":
        coords = sorted([int((l[0] + l[2]) / 2) for l in lines])
    else:
        coords = sorted([int((l[1] + l[3]) / 2) for l in lines])

    merged = []
    group = [coords[0]]

    for c in coords[1:]:
        if abs(c - group[-1]) <= gap_threshold:
            group.append(c)  # 距離夠近，歸入同一群組
        else:
            merged.append(int(sum(group) / len(group)))  # 取平均作為合併後座標
            group = [c]
    merged.append(int(sum(group) / len(group)))

    return merged

def detect_table_lines(image):
    """
    使用形態學運算偵測圖片中的表格水平線與垂直線
    
    Returns:
        merged_x:       垂直線的 X 座標清單
        merged_y:       水平線的 Y 座標清單
        horizontal_img: 水平線遮罩圖
        vertical_img:   垂直線遮罩圖
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)  # 二值化，深色線條變白色

    h, w = binary.shape

    # 水平線：長寬比大的矩形核
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 30), 1))
    # 垂直線：長高比大的矩形核
    vertical_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 30)))

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel,   iterations=1)

    horizontal_contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vertical_contours, _   = cv2.findContours(vertical,   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 過濾太短的線條（可能是雜訊）
    h_lines = []
    for cnt in horizontal_contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww > w * 0.15:  # 水平線長度須超過圖寬 15%
            h_lines.append((x, y, x + ww, y + hh))

    v_lines = []
    for cnt in vertical_contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if hh > h * 0.10:  # 垂直線高度須超過圖高 10%
            v_lines.append((x, y, x + ww, y + hh))

    # 合併相近線條
    merged_y = merge_nearby_lines(h_lines, axis="y", gap_threshold=12)
    merged_x = merge_nearby_lines(v_lines, axis="x", gap_threshold=12)

    return merged_x, merged_y, horizontal, vertical

def build_cells_from_lines(xs, ys, min_cell_w=20, min_cell_h=15):
    """
    根據水平線 Y 座標與垂直線 X 座標，建立表格 Cell 清單
    
    Args:
        xs:         垂直線 X 座標清單
        ys:         水平線 Y 座標清單
        min_cell_w: 最小 Cell 寬度（過濾雜訊）
        min_cell_h: 最小 Cell 高度（過濾雜訊）
    Returns:
        cells list: [{ row, col, bbox, texts, text }, ...]
    """
    cells = []
    if len(xs) < 2 or len(ys) < 2:
        return cells  # 線條不足，無法建立表格

    row_idx = 0
    for r in range(len(ys) - 1):
        col_idx = 0
        for c in range(len(xs) - 1):
            x1, x2 = xs[c], xs[c + 1]
            y1, y2 = ys[r], ys[r + 1]

            if (x2 - x1) >= min_cell_w and (y2 - y1) >= min_cell_h:
                cells.append({
                    "row":   row_idx,
                    "col":   col_idx,
                    "bbox":  [x1, y1, x2, y2],
                    "texts": [],
                    "text":  ""
                })
                col_idx += 1
        row_idx += 1

    return cells

def run_tsr(image):
    """
    對圖片執行完整表格結構辨識（偵測線條 → 建立 Cells）
    
    Returns:
        tsr_result:    { vertical_lines, horizontal_lines, cells }
        horizontal_img: 水平線遮罩
        vertical_img:   垂直線遮罩
    """
    xs, ys, horizontal_img, vertical_img = detect_table_lines(image)
    cells = build_cells_from_lines(xs, ys)
    tsr_result = {
        "vertical_lines":   xs,
        "horizontal_lines": ys,
        "cells":            cells
    }
    return tsr_result, horizontal_img, vertical_img

# =========================
# Cell Alignment（OCR 文字對齊到 Cell）
# =========================
def split_ocr_item_by_chars(item):
    """
    將一個 OCR item 拆成每個字元一個 item
    用於處理一段文字橫跨多個 Cell 的情況
    """
    text = item["text"]
    bbox = item["bbox"]

    if not text or len(text) <= 1:
        return [item]

    x1, y1, x2, y2 = bbox
    total_w    = x2 - x1
    char_count = len(text)

    if total_w <= 0 or char_count <= 1:
        return [item]

    char_w = total_w / char_count  # 每個字元平均寬度
    split_items = []

    for i, ch in enumerate(text):
        sx1 = int(round(x1 + i * char_w))
        sx2 = int(round(x1 + (i + 1) * char_w))
        split_items.append({
            "text": ch,
            "bbox": [sx1, y1, sx2, y2],
            "poly": [[sx1, y1], [sx2, y1], [sx2, y2], [sx1, y2]]
        })

    return split_items

def get_overlapping_cell_indices(item_bbox, cells, threshold=0.15):
    """找出與指定 bbox 重疊超過 threshold 的所有 Cell 索引"""
    matched = []
    for idx, cell in enumerate(cells):
        score = overlap_ratio(item_bbox, cell["bbox"])
        if score >= threshold:
            matched.append((idx, score))
    return matched

def align_ocr_to_cells(ocr_items, cells):
    """
    將 OCR items 對齊到 TSR 的 Cell 中
    
    策略：
    1. 若 OCR item 橫跨多個 Cell（overlap >= 2），拆成字元再對齊
    2. 先用 overlap_ratio 找最佳 Cell
    3. overlap 不夠時用中心點 fallback
    
    Returns:
        aligned_cells: 每個 Cell 附帶對齊的 OCR 文字
    """
    def merge_cell_texts(cell_texts):
        """將 Cell 內多個 OCR item 依閱讀順序合併成字串"""
        sorted_texts = sort_bbox_texts(cell_texts, y_tolerance=10)
        return "".join([t["text"] for t in sorted_texts]).strip()

    # 初始化對齊後的 Cell 清單
    aligned_cells = []
    for cell in cells:
        aligned_cells.append({
            "row":   cell["row"],
            "col":   cell["col"],
            "bbox":  cell["bbox"],
            "texts": [],
            "text":  ""
        })

    # 第一步：對跨格文字進行字元拆分
    expanded_items = []
    for item in ocr_items:
        overlaps = get_overlapping_cell_indices(item["bbox"], aligned_cells, threshold=0.15)
        if len(overlaps) >= 2 and len(item["text"]) > 1:
            expanded_items.extend(split_ocr_item_by_chars(item))  # 拆成字元
        else:
            expanded_items.append(item)

    # 第二步：將每個 item 對齊到最佳 Cell
    for item in expanded_items:
        bbox = item["bbox"]
        cx = (bbox[0] + bbox[2]) / 2  # 中心 X
        cy = (bbox[1] + bbox[3]) / 2  # 中心 Y

        best_idx   = None
        best_score = -1.0

        # 用 overlap_ratio 找重疊最高的 Cell
        for idx, cell in enumerate(aligned_cells):
            score = overlap_ratio(bbox, cell["bbox"])
            if score > best_score:
                best_score = score
                best_idx   = idx

        # overlap 太低時，改用中心點判斷
        if best_score < 0.3:
            for idx, cell in enumerate(aligned_cells):
                if point_in_bbox(cx, cy, cell["bbox"]):
                    best_idx = idx
                    break

        if best_idx is not None:
            aligned_cells[best_idx]["texts"].append(item)

    # 第三步：將 Cell 內文字排序後合併成字串
    for cell in aligned_cells:
        cell["texts"] = sort_bbox_texts(cell["texts"], y_tolerance=10)
        cell["text"]  = merge_cell_texts(cell["texts"])

    return aligned_cells

def build_table_from_cells(cells):
    """
    將 aligned cells 轉成二維表格（list of list）
    
    Returns:
        table: [ [row0col0, row0col1, ...], [row1col0, ...], ... ]
    """
    if not cells:
        return []

    max_row = max(cell["row"] for cell in cells)
    max_col = max(cell["col"] for cell in cells)

    table = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

    for cell in cells:
        table[cell["row"]][cell["col"]] = cell.get("text", "")

    return table

def safe_int(text):
    """將文字安全轉換成整數，失敗回傳 None"""
    if text is None:
        return None
    text = str(text).replace(",", "").strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    return None

def normalize_cell_text(text):
    """移除 Cell 文字中的所有空白"""
    if text is None:
        return ""
    return re.sub(r"\s+", "", str(text))

def find_header_row_and_columns(table_data):
    """
    在表格中找「品名、數量、單價、金額」的標頭行，並回傳各欄的 col index
    
    Returns:
        (header_row_idx, col_map) 或 (None, None)
    """
    for row_idx, row in enumerate(table_data):
        normalized_row = [normalize_cell_text(c) for c in row]

        col_map = {
            "品名": None,
            "數量": None,
            "單價": None,
            "金額": None
        }

        for col_idx, cell in enumerate(normalized_row):
            if "品名" in cell or cell in ["品名", "名品"]:
                col_map["品名"] = col_idx
            elif "數量" in cell:
                col_map["數量"] = col_idx
            elif "單價" in cell or "價單" in cell:
                col_map["單價"] = col_idx
            elif "金額" in cell or "額金" in cell:
                col_map["金額"] = col_idx

        # 四個欄位都找到才算找到標頭
        if all(v is not None for v in col_map.values()):
            return row_idx, col_map

    return None, None

# =========================
# 可視化
# =========================
def draw_ocr_boxes(img, ocr_items):
    """在圖片上畫出 OCR 文字框（綠色多邊形）"""
    for item in ocr_items:
        pts = np.array(item["poly"]).astype(int)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    return img

def draw_cells(img, cells):
    """在圖片上畫出 TSR Cell 邊框（藍色矩形）"""
    for cell in cells:
        x1, y1, x2, y2 = cell["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img

def draw_texts_with_pil(img, ocr_items, summary_lines, aligned_cells=None):
    """
    用 PIL 在圖片上疊加中文文字（OpenCV 不支援中文）
    
    Args:
        img:           原始圖片（BGR numpy array）
        ocr_items:     OCR 結果，用紅色標注在框框上方
        summary_lines: 頁面摘要文字，顯示在左上角（藍色）
        aligned_cells: 對齊後的 Cell，顯示對齊文字（橘色）
    Returns:
        疊加文字後的圖片（BGR numpy array）
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 紅色：OCR 辨識文字標注
    for item in ocr_items:
        x1, y1, x2, y2 = item["bbox"]
        draw.text(
            (x1, max(y1 - 22, 0)),
            item["text"],
            font=small_font,
            fill=(255, 0, 0)
        )

    # 橘色：Cell 對齊後的文字
    if aligned_cells:
        for cell in aligned_cells:
            if cell["text"]:
                x1, y1, x2, y2 = cell["bbox"]
                draw.text(
                    (x1 + 2, y1 + 2),
                    cell["text"],
                    font=small_font,
                    fill=(0, 128, 255)
                )

    # 藍色：左上角摘要資訊
    summary_text = "\n".join(summary_lines)
    draw.text((20, 20), summary_text, font=font, fill=(0, 0, 255))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =========================
# 金額檢核
# =========================
def validate_amounts(table_data):
    """
    從表格資料中檢核金額邏輯：
    1. 明細列：數量 × 單價 = 金額
    2. 明細金額加總 = 銷售額合計
    3. 銷售額合計 + 營業稅 = 總計
    
    Returns:
        dict: 各項檢核結果
    """
    result = {
        "detail_checks":             [],    # 每列明細的檢核結果
        "detail_sum":                None,  # 明細金額加總
        "sales_amount":              None,  # 銷售額合計
        "tax_amount":                None,  # 營業稅
        "total_amount":              None,  # 總計
        "detail_sum_match_sales":    None,  # 明細加總 == 銷售額合計？
        "sales_plus_tax_match_total": None, # 銷售額 + 稅 == 總計？
        "header_row_index":          None,  # 標頭行位置
        "detail_col_map":            None   # 各欄對應 col index
    }

    # 找標頭行
    header_row_idx, col_map = find_header_row_and_columns(table_data)
    result["header_row_index"] = header_row_idx
    result["detail_col_map"]   = col_map

    detail_sum   = 0
    detail_count = 0

    # 逐列處理明細資料（標頭行以下，遇到合計列就停止）
    if header_row_idx is not None and col_map is not None:
        for row in table_data[header_row_idx + 1:]:
            merged = "".join(normalize_cell_text(c) for c in row)

            # 遇到合計相關列，停止讀取明細
            if "銷售額合計" in merged or "營業稅" in merged or "營傢業稅" in merged or "總計" in merged:
                break

            max_idx = max(col_map.values())
            if len(row) <= max_idx:
                continue  # 欄數不足，跳過

            item_name  = normalize_cell_text(row[col_map["品名"]])
            qty        = safe_int(normalize_cell_text(row[col_map["數量"]]))
            unit_price = safe_int(normalize_cell_text(row[col_map["單價"]]))
            amount     = safe_int(normalize_cell_text(row[col_map["金額"]]))

            # 空白列跳過
            if not item_name and qty is None and unit_price is None and amount is None:
                continue

            if item_name and qty is not None and unit_price is not None and amount is not None:
                row_ok = (qty * unit_price == amount)  # 數量 × 單價 = 金額
                result["detail_checks"].append({
                    "品名": item_name,
                    "數量": qty,
                    "單價": unit_price,
                    "金額": amount,
                    "檢核結果": row_ok
                })
                detail_sum   += amount
                detail_count += 1

    if detail_count > 0:
        result["detail_sum"] = detail_sum

    # 掃全表找銷售額合計、營業稅、總計
    for row in table_data:
        merged = "".join(normalize_cell_text(c) for c in row)

        if "銷售額合計" in merged or "銷售額" in merged:
            for cell in row:
                val = safe_int(normalize_cell_text(cell))
                if val is not None:
                    result["sales_amount"] = val

        elif "營業稅" in merged or "營傢業稅" in merged or "稅" in merged:
            for cell in row:
                val = safe_int(normalize_cell_text(cell))
                if val is not None:
                    result["tax_amount"] = val

        elif "總計" in merged:
            for cell in row:
                val = safe_int(normalize_cell_text(cell))
                if val is not None:
                    result["total_amount"] = val

    # 最終邏輯驗證
    if result["detail_sum"] is not None and result["sales_amount"] is not None:
        result["detail_sum_match_sales"] = (result["detail_sum"] == result["sales_amount"])

    if (
        result["sales_amount"] is not None and
        result["tax_amount"]   is not None and
        result["total_amount"] is not None
    ):
        result["sales_plus_tax_match_total"] = (
            result["sales_amount"] + result["tax_amount"] == result["total_amount"]
        )

    return result

# =========================
# 載入 Excel 標準答案
# =========================
EXCEL_PATH = "file/會計憑證POC.xlsx"
standard_dict = load_excel_standard(EXCEL_PATH)

# =========================
# 主流程
# =========================
all_pages_result = []

for i, page in enumerate(pages):
    # --- 圖片轉換與儲存 ---
    img = np.array(page)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    jpg_path = f"jpg_pages/page_{i+1}_{pdf_name}.jpg"
    cv2.imwrite(jpg_path, img)
    print(f"已轉換: {jpg_path}")

    # ---------------------------------
    # 1. OCR branch：執行文字辨識
    # ---------------------------------
    ocr_items = run_ocr(jpg_path)

    # 將所有辨識文字合併成純文字字串
    page_text       = "\n".join([item["text"] for item in ocr_items])
    page_text_clean = clean_text(page_text)

    print(f"\n===== 第 {i+1} 頁 OCR 純文字 =====")
    print(page_text_clean)
    print("=================================\n")

    # 從 OCR 文字擷取關鍵欄位
    extracted_data = extract_fields_from_ocr_text(page_text_clean)

    # ---------------------------------
    # 2. TSR branch：表格結構辨識
    # ---------------------------------
    tsr_result, horizontal_img, vertical_img = run_tsr(img)
    cells = tsr_result["cells"]

    # ---------------------------------
    # 3. Alignment：OCR 文字對齊到 Cell
    # ---------------------------------
    aligned_cells = align_ocr_to_cells(ocr_items, cells)
    table_data    = build_table_from_cells(aligned_cells)

    # ---------------------------------
    # 3.5 金額檢核
    # ---------------------------------
    amount_validation = validate_amounts(table_data)

    # ---------------------------------
    # 4. Regex 檢核：統一編號驗證
    # ---------------------------------
    buyer_tax_id, seller_tax_id = extract_tax_id_by_context(page_text_clean)
    buyer_company_name = extract_buyer_company_name(page_text_clean)
    seller_company_name = extract_seller_company_name(page_text_clean)

    validation_result = {}
    validation_result["買方統一編號"]          = buyer_tax_id
    validation_result["買方公司名稱"]          = buyer_company_name
    validation_result["賣方統一編號"]          = seller_tax_id
    validation_result["賣方公司名稱"]    = seller_company_name
    validation_result["買方統編固定值檢核"]     = (buyer_tax_id == BUYER_TAX_ID_FIXED)  # 是否等於 05637971
    validation_result["統一編號格式正確"]       = bool(buyer_tax_id and re.fullmatch(r"\d{8}", buyer_tax_id))
    validation_result["金額檢核"]              = amount_validation
    validation_result["明細加總是否等於銷售額合計"]   = amount_validation["detail_sum_match_sales"]
    validation_result["銷售額合計加營業稅是否等於總計"] = amount_validation["sales_plus_tax_match_total"]
    

    # ---------------------------------
    # 4.5 與 Excel 標準答案比對
    # ---------------------------------
    extracted_invoice_no = extracted_data.get("發票號碼")
    lookup_invoice_no    = extracted_invoice_no

    standard = standard_dict.get(lookup_invoice_no)

    # 發票號碼找不到時，改用賣方統編從 Excel 反查
    if standard is None and seller_tax_id:
        for inv_no, std in standard_dict.items():
            if std.get("廠商統編") == seller_tax_id:
                standard          = std
                lookup_invoice_no = inv_no
                print(f"⚠️  發票號碼由 Excel 反查得到：{inv_no}（依賣方統編 {seller_tax_id} 比對）")
                break

    if standard is None:
        print(f"⚠️  找不到發票號碼 [{lookup_invoice_no}] 的標準答案")

    compare_result = compare_with_standard(
        buyer_tax_id,
        seller_tax_id,
        buyer_company_name,
        seller_company_name,
        amount_validation,
        extracted_invoice_no,
        standard
    )
    validation_result["與Excel比對結果"] = compare_result

    # ---------------------------------
    # 5. 顯示結果
    # ---------------------------------
    print(f"第 {i+1} 頁擷取結果:")
    for k, v in extracted_data.items():
        print(f"  {k}: {v}")

    print(f"\n第 {i+1} 頁檢核結果:")
    for k, v in validation_result.items():
        if k != "金額檢核":  # 金額檢核細節太多，不在此印出
            print(f"  {k}: {v}")

    print(f"\n第 {i+1} 頁 與標準答案比對:")
    for k, v in compare_result.items():
        print(f"  {k}: {v}")

    print(f"\n第 {i+1} 頁 TSR cells 數量: {len(cells)}")
    print("-" * 50)

    # ---------------------------------
    # 6. 視覺化輸出
    # ---------------------------------
    vis_img = img.copy()
    vis_img = draw_ocr_boxes(vis_img, ocr_items)   # 畫 OCR 框（綠）
    vis_img = draw_cells(vis_img, aligned_cells)    # 畫 Cell 框（藍）

    # 左上角摘要資訊
    summary_lines = [
        f"Page: {i+1}",
        f"發票號碼: {extracted_invoice_no or '未找到'}",
        f"買方統編: {buyer_tax_id or '未找到'} ({'✓' if validation_result['買方統編固定值檢核'] else '✗'})",
        f"賣方統編: {seller_tax_id or '未找到'}",
        f"統一編號格式正確: {validation_result['統一編號格式正確']}",
        f"Excel比對全部通過: {compare_result.get('全部比對通過', 'N/A')}",
        f"OCR文字框數: {len(ocr_items)}",
        f"TSR cells數: {len(cells)}"
    ]

    final_img = draw_texts_with_pil(
        vis_img,
        ocr_items,
        summary_lines,
        aligned_cells=aligned_cells
    )

    # 儲存視覺化結果
    save_path = f"output/page_{i+1}_{pdf_name}.jpg"
    cv2.imwrite(save_path, final_img)
    print(f"輸出: {save_path}")

    # 儲存水平線與垂直線遮罩（Debug 用）
    cv2.imwrite(f"output/page_{i+1}_{pdf_name}_horizontal.jpg", horizontal_img)
    cv2.imwrite(f"output/page_{i+1}_{pdf_name}_vertical.jpg",   vertical_img)

    # ---------------------------------
    # 7. JSON 輸出（單頁）
    # ---------------------------------
    page_result = {
        "page":           i + 1,
        "ocr_text":       page_text,
        "ocr_text_clean": page_text_clean,
        "ocr_items":      ocr_items,
        "extracted_data": extracted_data,
        "validation_result": validation_result,
        "compare_result": compare_result,
        "tsr_result":     tsr_result,
        "aligned_cells":  aligned_cells,
        "table_data":     table_data
    }

    all_pages_result.append(page_result)

    with open(f"json_output/page_{i+1}.json", "w", encoding="utf-8") as f:
        json.dump(page_result, f, ensure_ascii=False, indent=2)

# =========================
# 全部頁面總結果輸出
# =========================
with open("json_output/all_pages_result.json", "w", encoding="utf-8") as f:
    json.dump(all_pages_result, f, ensure_ascii=False, indent=2)

print("完成")