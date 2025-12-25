# -*- coding: utf-8 -*-
import os
import sys
# 讓它可以找到同目錄下的模組
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import institutional  # 匯入法人資料模組
import time
import random
import requests
import pandas as pd
import yfinance as yf
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

# ========== 核心參數設定 ==========
START_DATE = "2024-01-01"  # 歷史資料起始日期
MARKET_CODE = "tw-share"
DATA_SUBDIR = "dayK"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", MARKET_CODE, DATA_SUBDIR)

# ✅ 效能優化：調低至 2-3，配合亂數延遲可有效避開 Yahoo 封鎖
MAX_WORKERS = 3 
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"{pd.Timestamp.now():%H:%M:%S}: {msg}")

def get_full_stock_list():
    """獲取台股全市場清單 (排除權證)"""
    url_configs = [
        {'name': 'listed', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?market=1&issuetype=1&Page=1&chklike=Y', 'suffix': '.TW'},
        {'name': 'dr', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=1&issuetype=J&industry_code=&Page=1&chklike=Y', 'suffix': '.TW'},
        {'name': 'otc', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?market=2&issuetype=4&Page=1&chklike=Y', 'suffix': '.TWO'},
        {'name': 'etf', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=1&issuetype=I&industry_code=&Page=1&chklike=Y', 'suffix': '.TW'},
        {'name': 'rotc', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=E&issuetype=R&industry_code=&Page=1&chklike=Y', 'suffix': '.TWO'},
        {'name': 'tw_innovation', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=C&issuetype=C&industry_code=&Page=1&chklike=Y', 'suffix': '.TW'},
        {'name': 'otc_innovation', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=A&issuetype=C&industry_code=&Page=1&chklike=Y', 'suffix': '.TWO'},
    ]
    all_items = []
    log("📡 正在獲取各市場清單...")
    for cfg in url_configs:
        try:
            resp = requests.get(cfg['url'], timeout=15)
            df_list = pd.read_html(StringIO(resp.text), header=0)
            if not df_list: continue
            df = df_list[0]
            for _, row in df.iterrows():
                code = str(row['有價證券代號']).strip()
                name = str(row['有價證券名稱']).strip()
                if code and '有價證券' not in code:
                    all_items.append(f"{code}{cfg['suffix']}&{name}")
        except: continue
    return list(set(all_items))

def download_stock_data(item):
    """具備增量更新與隨機延遲的下載邏輯"""
    yf_tkr = "ParseError"
    try:
        parts = item.split('&', 1)
        if len(parts) < 2: return {"status": "error", "tkr": item, "msg": "Format error"}
        
        yf_tkr, name = parts
        # 移除檔名非法字元
        safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '_', '-')]).strip()
        out_path = os.path.join(DATA_DIR, f"{yf_tkr}_{safe_name}.csv")
        
        # ========== 增量更新邏輯 ==========
        existing_df = None
        start_date = START_DATE
        
        if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
            try:
                existing_df = pd.read_csv(out_path)
                existing_df['date'] = pd.to_datetime(existing_df['date']).dt.tz_localize(None)
                last_date = existing_df['date'].max()
                
                # 取得今天日期（不含時間）
                today = pd.Timestamp.now().normalize()
                last_date_normalized = last_date.normalize()
                
                # 如果最後日期 >= 今天，視為已更新
                if last_date_normalized >= today:
                    return {"status": "exists", "tkr": yf_tkr}
                
                # 從最後日期的下一天開始下載
                start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            except Exception:
                existing_df = None
                start_date = START_DATE


        # ✅ 關鍵 1: 初始隨機休眠 (0.5~1.15秒)，打亂請求頻率
        time.sleep(random.uniform(0.5, 1.15))

        tk = yf.Ticker(yf_tkr)
        
        # ✅ 關鍵 2: 雙重重試機制
        for attempt in range(2):
            try:
                hist = tk.history(start=start_date, timeout=15)
                if hist is not None and not hist.empty:
                    hist.reset_index(inplace=True)
                    hist.columns = [c.lower() for c in hist.columns]
                    
                    # 如果有舊資料，合併新舊資料
                    if existing_df is not None:
                        hist['date'] = pd.to_datetime(hist['date']).dt.tz_localize(None)
                        combined_df = pd.concat([existing_df, hist], ignore_index=True)
                        combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                        combined_df = combined_df.sort_values('date').reset_index(drop=True)
                        combined_df.to_csv(out_path, index=False, encoding='utf-8-sig')
                        return {"status": "updated", "tkr": yf_tkr}
                    else:
                        hist.to_csv(out_path, index=False, encoding='utf-8-sig')
                        return {"status": "success", "tkr": yf_tkr}
                
                # 如果是 Empty，可能是該代號真的沒資料
                if attempt == 1: return {"status": "empty", "tkr": yf_tkr}
                
            except Exception as e:
                # 如果遇到 Rate Limit，休眠時間加長
                if "Rate limited" in str(e):
                    time.sleep(random.uniform(15, 30))
                if attempt == 1: return {"status": "error", "tkr": yf_tkr, "msg": str(e)}
            
            # 重試前的隨機長休眠
            time.sleep(random.uniform(3, 7))

        return {"status": "empty", "tkr": yf_tkr}
    except Exception as e:
        return {"status": "error", "tkr": yf_tkr, "msg": str(e)}

def main():
    items = get_full_stock_list()
    log(f"🚀 啟動增量更新模式，目標總數: {len(items)}")
    log(f"📅 資料起始日期: {START_DATE}")
    
    stats = {"success": 0, "updated": 0, "exists": 0, "empty": 0, "error": 0}
    error_details = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_stock_data, it): it for it in items}
        pbar = tqdm(total=len(items), desc="下載進度")
        
        for future in as_completed(futures):
            res = future.result()
            s = res["status"]
            stats[s] += 1
            if s == "error":
                msg = res.get("msg", "Unknown Error")[:50]
                error_details[msg] = error_details.get(msg, 0) + 1
            pbar.update(1)
            
            # ✅ 每 50 檔輸出一次文字進度（給 Web UI 讀取）
            if pbar.n % 50 == 0 or pbar.n == len(items):
                done = stats['success'] + stats['updated'] + stats['exists']
                print(f"📊 進度: {pbar.n}/{len(items)} ({pbar.n*100//len(items)}%) | 完成:{done} 失敗:{stats['error']}", flush=True)
            
            # ✅ 額外保險：每下載 100 檔強制休息，清理連線
            if pbar.n % 100 == 0:
                time.sleep(random.uniform(5, 10))
                
        pbar.close()
    
    print("\n" + "="*50)
    log("📊 下載報告:")
    print(f"   - ✅ 新檔下載: {stats['success']}")
    print(f"   - 🔄 增量更新: {stats['updated']}")
    print(f"   - 📁 已是最新: {stats['exists']}")
    print(f"   - 🔍 Yahoo無資料: {stats['empty']}")
    print(f"   - ❌ 失敗: {stats['error']}")
    if error_details:
        print("\n⚠️ 失敗原因分析:")
        for msg, count in sorted(error_details.items(), key=lambda x: x[1], reverse=True):
            print(f"   - [{count}次]: {msg}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
