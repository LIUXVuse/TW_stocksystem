# -*- coding: utf-8 -*-
"""
融資融券資料抓取模組
從台灣證券交易所抓取融資融券餘額資料
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# 證交所 API URL
TWSE_MARGIN_URL = "https://www.twse.com.tw/exchangeReport/MI_MARGN"

def get_margin_data(date=None):
    """
    取得指定日期的融資融券餘額資料
    
    Args:
        date: 日期 (YYYYMMDD 格式)，預設為今天
    
    Returns:
        dict: {
            ticker: {
                margin_buy: int,      # 融資買進
                margin_sell: int,     # 融資賣出
                margin_balance: int,  # 融資餘額
                margin_limit: int,    # 融資限額
                margin_use_rate: float, # 融資使用率
                short_buy: int,       # 融券買進
                short_sell: int,      # 融券賣出
                short_balance: int,   # 融券餘額
            }
        }
    """
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    result = {}
    
    try:
        # 上市股票融資融券
        url = f"{TWSE_MARGIN_URL}?response=json&date={date}&selectType=ALL"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        data = response.json()
        
        # 證交所融資融券 API 回傳的是 tables 陣列
        # tables[0] 是市場總計，tables[1] 是個股明細
        if 'tables' in data and len(data['tables']) > 1:
            stock_data = data['tables'][1].get('data', [])
            for row in stock_data:
                try:
                    # 欄位順序：
                    # 0: 股票代號, 1: 股票名稱
                    # 2: 融資買進, 3: 融資賣出, 4: 融資現金償還, 5: 融資前日餘額
                    # 6: 融資今日餘額, 7: 融資限額, 8: 融資使用率
                    # 9: 融券賣出, 10: 融券買進, 11: 融券現券償還, 12: 融券前日餘額
                    # 13: 融券今日餘額, 14: 資券互抵
                    
                    ticker = row[0].strip()
                    
                    def parse_int(val):
                        if isinstance(val, (int, float)):
                            return int(val)
                        return int(str(val).replace(',', '').replace(' ', '') or 0)
                    
                    def parse_float(val):
                        if isinstance(val, (int, float)):
                            return float(val)
                        return float(str(val).replace(',', '').replace('%', '').replace(' ', '') or 0)
                    
                    result[ticker + '.TW'] = {
                        'margin_buy': parse_int(row[2]),
                        'margin_sell': parse_int(row[3]),
                        'margin_balance': parse_int(row[6]),
                        'margin_limit': parse_int(row[7]),
                        'margin_use_rate': parse_float(row[8]),
                        'short_buy': parse_int(row[10]),
                        'short_sell': parse_int(row[9]),
                        'short_balance': parse_int(row[13]),
                    }
                except:
                    continue
    except Exception as e:
        print(f"抓取融資融券失敗: {e}")
    
    return result

def get_margin_summary():
    """
    取得融資融券市場概況
    """
    today = datetime.now().strftime('%Y%m%d')
    
    try:
        url = f"https://www.twse.com.tw/exchangeReport/MI_MARGN?response=json&date={today}&selectType=MS"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        data = response.json()
        
        if 'creditList' in data:
            credit = data['creditList'][0] if data['creditList'] else None
            if credit:
                return {
                    'margin_buy': credit.get('融資買進', 0),
                    'margin_sell': credit.get('融資賣出', 0),
                    'margin_balance': credit.get('融資餘額', 0),
                    'short_sell': credit.get('融券賣出', 0),
                    'short_buy': credit.get('融券買進', 0),
                    'short_balance': credit.get('融券餘額', 0),
                }
    except Exception as e:
        print(f"抓取融資融券概況失敗: {e}")
    
    return None

def analyze_margin_sentiment(margin_data):
    """
    分析融資融券情緒
    
    Returns:
        dict: {
            ticker: {
                margin_change: int,   # 融資增減
                short_change: int,    # 融券增減
                sentiment: str,       # 市場情緒 (bullish/bearish/neutral)
            }
        }
    """
    result = {}
    
    for ticker, data in margin_data.items():
        margin_net = data['margin_buy'] - data['margin_sell']  # 融資淨買
        short_net = data['short_sell'] - data['short_buy']     # 融券淨賣
        
        # 判斷情緒
        if margin_net > 0 and short_net < 0:
            sentiment = 'bullish'   # 散戶看多
        elif margin_net < 0 and short_net > 0:
            sentiment = 'bearish'   # 散戶看空
        else:
            sentiment = 'neutral'
        
        result[ticker] = {
            'margin_change': margin_net,
            'short_change': short_net,
            'margin_use_rate': data.get('margin_use_rate', 0),
            'sentiment': sentiment,
        }
    
    return result


def download_margin_history(start_date, end_date=None, save_dir=None):
    """
    批次下載融資融券歷史資料
    
    Args:
        start_date: 起始日期 (YYYYMMDD)
        end_date: 結束日期 (YYYYMMDD)，預設為今天
        save_dir: 儲存目錄，預設為 data/margin/
    
    Returns:
        dict: {date: {ticker: {...}}}
    """
    import os
    import json
    from pathlib import Path
    from tqdm import tqdm
    from datetime import timedelta
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    if save_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, "data", "margin")
    
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
    
    print(f"💰 下載融資融券歷史資料")
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
            data = get_margin_data(date)
            
            if data:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                success += 1
                print(f"📊 融資融券進度: {current}/{total} ({current*100//total}%) | {date} 成功", flush=True)
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


def auto_update():
    """
    自動更新融資融券資料
    - 檢查 data/margin/ 下最新的檔案
    - 如果沒有資料，從 20240101 開始抓取
    - 如果有資料，從最新日期的下一天開始抓取
    """
    import os
    from glob import glob
    from datetime import datetime, timedelta
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "data", "margin")
    
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
        print(f"✅ 融資融券資料已是最新 ({end_date})")
        return
        
    print(f"🔄 自動更新融資融券資料: {start_date} -> {end_date}")
    download_margin_history(start_date, end_date, save_dir)


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
            download_margin_history(start, end)
    else:
        # 預設測試
        print("測試抓取融資融券資料...")
        data = get_margin_data()
        print(f"共抓取 {len(data)} 支股票")
        
        # 顯示前 5 筆
        for i, (ticker, values) in enumerate(list(data.items())[:5]):
            print(f"{ticker}: 融資餘額 {values['margin_balance']:,} 融券餘額 {values['short_balance']:,}")
        
        print()
        print("💡 下載用法:")
        print("   python margin.py auto              (自動更新)")
        print("   python margin.py 20240101 20241220 (手動指定)")
