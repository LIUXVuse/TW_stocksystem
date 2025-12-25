# -*- coding: utf-8 -*-
"""
資料載入模組
整合股價、法人、融資融券資料
"""
import os
import json
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

# ========== 路徑設定 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOCK_DIR = os.path.join(BASE_DIR, "data", "tw-share", "dayK")
INSTITUTIONAL_DIR = os.path.join(BASE_DIR, "data", "institutional")
MARGIN_DIR = os.path.join(BASE_DIR, "data", "margin")


def find_stock_file(ticker: str) -> str:
    """根據股票代碼找到對應的 CSV 檔案"""
    # 嘗試直接匹配
    pattern = os.path.join(STOCK_DIR, f"{ticker}_*.csv")
    matches = glob(pattern)
    if matches:
        return matches[0]
    
    # 嘗試不同格式
    pattern = os.path.join(STOCK_DIR, f"*{ticker}*.csv")
    matches = glob(pattern)
    if matches:
        return matches[0]
    
    return None


def load_institutional_data() -> dict:
    """
    載入所有法人歷史資料
    
    Returns:
        dict: {date_str: {ticker: {foreign, trust, dealer, total}}}
    """
    all_data = {}
    files = sorted(glob(os.path.join(INSTITUTIONAL_DIR, "*.json")))
    
    for f in files:
        date_str = os.path.basename(f).replace('.json', '')
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                all_data[date_str] = json.load(fp)
        except:
            continue
    
    return all_data


def load_margin_data() -> dict:
    """
    載入所有融資融券歷史資料
    
    Returns:
        dict: {date_str: {ticker: {margin_balance, short_balance, ...}}}
    """
    all_data = {}
    files = sorted(glob(os.path.join(MARGIN_DIR, "*.json")))
    
    for f in files:
        date_str = os.path.basename(f).replace('.json', '')
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                all_data[date_str] = json.load(fp)
        except:
            continue
    
    return all_data


def load_stock_with_institutional(ticker: str, 
                                   include_margin: bool = False) -> pd.DataFrame:
    """
    載入股票資料並合併法人資料
    
    Args:
        ticker: 股票代碼（如 2330.TW）
        include_margin: 是否包含融資融券資料
    
    Returns:
        DataFrame: 包含 OHLCV + 技術指標 + 法人資料
    """
    # 載入股價資料
    csv_path = find_stock_file(ticker)
    if not csv_path:
        raise FileNotFoundError(f"找不到股票 {ticker} 的資料檔案")
    
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    
    # 處理日期
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df['date_str'] = df['date'].dt.strftime('%Y%m%d')
    
    # 載入法人資料
    inst_data = load_institutional_data()
    
    # 初始化法人欄位
    df['foreign'] = 0
    df['trust'] = 0
    df['dealer'] = 0
    df['inst_total'] = 0
    
    # 合併法人資料
    for i, row in df.iterrows():
        date_str = row['date_str']
        if date_str in inst_data:
            stock_data = inst_data[date_str].get(ticker, {})
            df.at[i, 'foreign'] = stock_data.get('foreign', 0)
            df.at[i, 'trust'] = stock_data.get('trust', 0)
            df.at[i, 'dealer'] = stock_data.get('dealer', 0)
            df.at[i, 'inst_total'] = stock_data.get('total', 0)
    
    # 計算法人累計
    df['foreign_5d'] = df['foreign'].rolling(5).sum()
    df['trust_5d'] = df['trust'].rolling(5).sum()
    df['inst_5d'] = df['inst_total'].rolling(5).sum()
    
    # 融資融券資料（可選）
    if include_margin:
        margin_data = load_margin_data()
        
        df['margin_balance'] = 0
        df['short_balance'] = 0
        df['margin_change'] = 0
        df['short_change'] = 0
        
        prev_margin = 0
        prev_short = 0
        
        for i, row in df.iterrows():
            date_str = row['date_str']
            if date_str in margin_data:
                stock_data = margin_data[date_str].get(ticker, {})
                curr_margin = stock_data.get('margin_balance', 0)
                curr_short = stock_data.get('short_balance', 0)
                
                df.at[i, 'margin_balance'] = curr_margin
                df.at[i, 'short_balance'] = curr_short
                df.at[i, 'margin_change'] = curr_margin - prev_margin
                df.at[i, 'short_change'] = curr_short - prev_short
                
                prev_margin = curr_margin
                prev_short = curr_short
    
    # 移除暫存欄位
    df = df.drop(columns=['date_str'])
    
    return df


def get_all_tickers() -> list:
    """取得所有可用的股票代碼"""
    files = glob(os.path.join(STOCK_DIR, "*.csv"))
    tickers = []
    for f in files:
        basename = os.path.basename(f)
        ticker = basename.split('_')[0]
        tickers.append(ticker)
    return sorted(tickers)


def get_popular_tickers() -> list:
    """取得熱門股票代碼"""
    return [
        '2330.TW',   # 台積電
        '2317.TW',   # 鴻海
        '2454.TW',   # 聯發科
        '2308.TW',   # 台達電
        '2881.TW',   # 富邦金
        '2882.TW',   # 國泰金
        '2412.TW',   # 中華電
        '1301.TW',   # 台塑
        '2002.TW',   # 中鋼
        '3711.TW',   # 日月光投控
        '0050.TW',   # 元大台灣50
        '0056.TW',   # 元大高股息
        '00878.TW',  # 國泰永續高股息
    ]


if __name__ == '__main__':
    # 測試
    print("測試載入台積電資料（含法人）...")
    df = load_stock_with_institutional('2330.TW', include_margin=True)
    
    print(f"\n總欄位數: {len(df.columns)}")
    print(f"資料筆數: {len(df)}")
    
    print("\n新增的法人欄位:")
    inst_cols = ['foreign', 'trust', 'dealer', 'inst_total', 'foreign_5d', 'trust_5d', 'inst_5d']
    for c in inst_cols:
        if c in df.columns:
            print(f"  • {c}")
    
    print("\n最後 5 筆資料:")
    print(df[['date', 'close', 'foreign', 'trust', 'inst_total', 'foreign_5d']].tail().to_string())
