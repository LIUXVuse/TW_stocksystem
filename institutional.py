# -*- coding: utf-8 -*-
"""
三大法人買賣超資料抓取模組
從台灣證券交易所抓取外資、投信、自營商買賣超資料
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# 證交所 API URL
TWSE_INSTITUTIONAL_URL = "https://www.twse.com.tw/fund/T86"
TPEX_INSTITUTIONAL_URL = "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"

def get_institutional_data(date=None):
    """
    取得指定日期的三大法人買賣超資料
    
    Args:
        date: 日期 (YYYYMMDD 格式)，預設為今天
    
    Returns:
        dict: {ticker: {foreign: int, trust: int, dealer: int, total: int}}
    """
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    result = {}
    
    # 抓取上市股票（TWSE）
    try:
        twse_data = fetch_twse_institutional(date)
        result.update(twse_data)
    except Exception as e:
        print(f"抓取上市三大法人失敗: {e}")
    
    # 抓取上櫃股票（TPEx）
    try:
        tpex_data = fetch_tpex_institutional(date)
        result.update(tpex_data)
    except Exception as e:
        print(f"抓取上櫃三大法人失敗: {e}")
    
    return result

def fetch_twse_institutional(date):
    """抓取上市股票三大法人資料"""
    url = f"{TWSE_INSTITUTIONAL_URL}?response=json&date={date}&selectType=ALLBUT0999"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    data = response.json()
    
    result = {}
    
    if 'data' in data:
        for row in data['data']:
            try:
                # 欄位定義 (證交所最新格式 - 19 欄):
                # [0] 證券代號
                # [1] 證券名稱
                # [2] 外陸資買進股數(不含外資自營商)
                # [3] 外陸資賣出股數(不含外資自營商)
                # [4] 外陸資買賣超股數(不含外資自營商)
                # [5] 外資自營商買進股數
                # [6] 外資自營商賣出股數
                # [7] 外資自營商買賣超股數
                # [8] 投信買進股數
                # [9] 投信賣出股數
                # [10] 投信買賣超股數
                # [11] 自營商買賣超股數
                # [12-17] 自營商細項
                # [18] 三大法人買賣超股數
                
                ticker = row[0].strip()
                
                # 移除逗號並轉換為整數
                def parse_int(val):
                    return int(str(val).replace(',', '').replace(' ', ''))
                
                # 外資 = 外陸資 + 外資自營商
                foreign = parse_int(row[4]) + parse_int(row[7])
                
                result[ticker + '.TW'] = {
                    'foreign': foreign,                 # 外資買賣超（含自營商）
                    'trust': parse_int(row[10]),        # 投信買賣超
                    'dealer': parse_int(row[11]),       # 自營商買賣超
                    'total': parse_int(row[18])         # 三大法人合計
                }
            except:
                continue
    
    return result

def fetch_tpex_institutional(date):
    """抓取上櫃股票三大法人資料"""
    # 轉換日期格式 YYYYMMDD -> YYY/MM/DD (民國年)
    year = int(date[:4]) - 1911
    month = date[4:6]
    day = date[6:8]
    roc_date = f"{year}/{month}/{day}"
    
    url = f"{TPEX_INSTITUTIONAL_URL}?l=zh-tw&d={roc_date}&se=EW&t=D"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    data = response.json()
    
    result = {}
    
    if 'aaData' in data:
        for row in data['aaData']:
            try:
                ticker = row[0].strip()
                
                def parse_int(val):
                    return int(float(str(val).replace(',', '').replace(' ', '') or 0))
                
                result[ticker + '.TWO'] = {
                    'foreign': parse_int(row[4]),      # 外資買賣超
                    'trust': parse_int(row[7]),        # 投信買賣超
                    'dealer': parse_int(row[10]),      # 自營商買賣超
                    'total': parse_int(row[11])        # 三大法人合計
                }
            except:
                continue
    
    return result

def get_recent_institutional(days=5):
    """
    取得最近 N 天的三大法人累計買賣超
    
    Args:
        days: 天數
    
    Returns:
        dict: {ticker: {foreign: int, trust: int, dealer: int}}
    """
    cumulative = {}
    current_date = datetime.now()
    
    for i in range(days + 10):  # 多取幾天避免遇到假日
        date = (current_date - timedelta(days=i)).strftime('%Y%m%d')
        
        try:
            daily_data = get_institutional_data(date)
            
            if daily_data:
                for ticker, values in daily_data.items():
                    if ticker not in cumulative:
                        cumulative[ticker] = {'foreign': 0, 'trust': 0, 'dealer': 0, 'days': 0}
                    
                    cumulative[ticker]['foreign'] += values['foreign']
                    cumulative[ticker]['trust'] += values['trust']
                    cumulative[ticker]['dealer'] += values['dealer']
                    cumulative[ticker]['days'] += 1
                
                # 檢查是否已收集足夠天數
                if any(v['days'] >= days for v in cumulative.values()):
                    break
            
            time.sleep(0.5)  # 避免請求過快
            
        except Exception as e:
            print(f"取得 {date} 資料失敗: {e}")
            continue
    
    return cumulative


def download_institutional_history(start_date, end_date=None, save_dir=None):
    """
    批次下載法人歷史資料
    
    Args:
        start_date: 起始日期 (YYYYMMDD)
        end_date: 結束日期 (YYYYMMDD)，預設為今天
        save_dir: 儲存目錄，預設為 data/institutional/
    
    Returns:
        dict: {date: {ticker: {...}}}
    """
    import os
    import json
    from pathlib import Path
    from tqdm import tqdm
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    if save_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, "data", "institutional")
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 產生日期清單
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    
    dates = []
    current = start
    while current <= end:
        # 跳過週末
        if current.weekday() < 5:
            dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    
    print(f"📊 下載法人歷史資料")
    print(f"📅 日期範圍: {start_date} ~ {end_date}")
    print(f"📁 儲存位置: {save_dir}")
    print(f"📋 預計下載: {len(dates)} 天")
    print()
    
    success = 0
    skipped = 0
    failed = 0
    total = len(dates)
    current = 0
    
    for date in tqdm(dates, desc="下載進度"):
        file_path = os.path.join(save_dir, f"{date}.json")
        current += 1
        
        # 如果檔案已存在，跳過
        if os.path.exists(file_path):
            skipped += 1
            continue
        
        try:
            data = get_institutional_data(date)
            
            if data:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                success += 1
                print(f"📊 法人進度: {current}/{total} ({current*100//total}%) | {date} 成功", flush=True)
            else:
                failed += 1
            
            time.sleep(0.5)  # 避免請求過快
            
        except Exception as e:
            failed += 1
            continue
    
    print()
    print("=" * 50)
    print(f"✅ 成功下載: {success}")
    print(f"📁 已存在跳過: {skipped}")
    print(f"❌ 失敗/無資料: {failed}")
    print("=" * 50)


    print("=" * 50)


def auto_update():
    """
    自動更新法人資料
    - 檢查 data/institutional/ 下最新的檔案
    - 如果沒有資料，從 20240101 開始抓取
    - 如果有資料，從最新日期的**下一天**開始抓取
    - 抓取到**今天**
    """
    import os
    from glob import glob
    from datetime import datetime, timedelta
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "data", "institutional")
    
    # 確保目錄存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 找現有檔案
    files = glob(os.path.join(save_dir, "*.json"))
    
    start_date = "20240101"  # 預設起始日
    
    if files:
        # 找出最新日期
        dates = [os.path.splitext(os.path.basename(f))[0] for f in files]
        dates.sort()
        last_date = dates[-1]
        
        # 從下一天開始
        last_dt = datetime.strptime(last_date, '%Y%m%d')
        start_date = (last_dt + timedelta(days=1)).strftime('%Y%m%d')
        
    end_date = datetime.now().strftime('%Y%m%d')
    
    # 如果起始日已經晚於結束日，代表不用更新
    if start_date > end_date:
        print(f"✅ 法人資料已是最新 ({end_date})")
        return
        
    print(f"🔄 自動更新法人資料: {start_date} -> {end_date}")
    download_institutional_history(start_date, end_date, save_dir)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'auto':
            # 自動更新模式
            auto_update()
        else:
            # 手動指定日期模式
            start = sys.argv[1]
            end = sys.argv[2] if len(sys.argv) > 2 else None
            download_institutional_history(start, end)
    else:
        # 預設測試
        print("測試抓取今日三大法人資料...")
        data = get_institutional_data()
        print(f"共抓取 {len(data)} 支股票")
        
        # 顯示前 5 筆
        for i, (ticker, values) in enumerate(list(data.items())[:5]):
            print(f"{ticker}: 外資 {values['foreign']:+,} 投信 {values['trust']:+,} 自營 {values['dealer']:+,}")
        
        print()
        print("💡 下載歷史資料用法:")
        print("   python institutional.py auto              (自動更新)")
        print("   python institutional.py 20240101 20241224 (手動指定範圍)")

