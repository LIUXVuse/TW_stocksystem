# -*- coding: utf-8 -*-
"""
技術指標計算模組
計算常用技術分析指標並加入 CSV 檔案
"""
import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

# ========== 資料路徑設定 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "tw-share", "dayK")


# ========== 趨勢指標 ==========

def calc_ma(df, periods=[5, 10, 20, 60]):
    """計算移動平均線"""
    for p in periods:
        df[f'ma{p}'] = df['close'].rolling(window=p).mean()
    return df


def calc_ema(df, periods=[12, 26]):
    """計算指數移動平均線"""
    for p in periods:
        df[f'ema{p}'] = df['close'].ewm(span=p, adjust=False).mean()
    return df


def calc_macd(df, fast=12, slow=26, signal=9):
    """計算 MACD 指標"""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df


def calc_bollinger(df, period=20, std_dev=2):
    """計算布林通道"""
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    rolling_std = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
    df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
    return df


# ========== 動能指標 ==========

def calc_rsi(df, period=14):
    """計算 RSI 相對強弱指標"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df


def calc_kd(df, k_period=9, d_period=3):
    """計算 KD 隨機指標"""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    df['k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['d'] = df['k'].rolling(window=d_period).mean()
    return df


def calc_williams_r(df, period=14):
    """計算威廉指標 Williams %R"""
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    
    df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
    return df


# ========== 成交量指標 ==========

def calc_obv(df):
    """計算 OBV 能量潮"""
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    return df


def calc_volume_ma(df, periods=[5, 20]):
    """計算成交量均線"""
    for p in periods:
        df[f'vol_ma{p}'] = df['volume'].rolling(window=p).mean()
    return df


# ========== 波動率指標 ==========

def calc_atr(df, period=14):
    """計算 ATR 真實波動幅度均值"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=period).mean()
    return df


# ========== 主要函數 ==========

def calculate_all_indicators(df):
    """
    計算所有技術指標
    
    Args:
        df: 包含 OHLCV 資料的 DataFrame
    
    Returns:
        df: 加入所有指標欄位的 DataFrame
    """
    # 確保欄位名稱為小寫
    df.columns = [c.lower() for c in df.columns]
    
    # 趨勢指標
    df = calc_ma(df)
    df = calc_ema(df)
    df = calc_macd(df)
    df = calc_bollinger(df)
    
    # 動能指標
    df = calc_rsi(df)
    df = calc_kd(df)
    df = calc_williams_r(df)
    
    # 成交量指標
    df = calc_obv(df)
    df = calc_volume_ma(df)
    
    # 波動率指標
    df = calc_atr(df)
    
    return df


def add_indicators_to_csv(csv_path):
    """
    讀取 CSV，計算指標，覆寫回原檔案
    
    Args:
        csv_path: CSV 檔案路徑
    
    Returns:
        bool: 是否成功
    """
    try:
        df = pd.read_csv(csv_path)
        df = calculate_all_indicators(df)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        return True
    except Exception as e:
        print(f"處理失敗 {csv_path}: {e}")
        return False


def process_all_stocks():
    """批次處理所有股票的 CSV 檔案"""
    files = glob(os.path.join(DATA_DIR, "*.csv"))
    
    print(f"📊 開始計算技術指標...")
    print(f"📁 共 {len(files)} 個檔案")
    
    success = 0
    failed = 0
    
    for f in tqdm(files, desc="計算進度"):
        if add_indicators_to_csv(f):
            success += 1
        else:
            failed += 1
    
    print()
    print("=" * 50)
    print(f"✅ 成功: {success}")
    print(f"❌ 失敗: {failed}")
    print("=" * 50)
    
    # 顯示新增的欄位
    if files:
        sample = pd.read_csv(files[0])
        new_cols = [c for c in sample.columns if c not in ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits']]
        print(f"\n📋 新增的技術指標欄位 ({len(new_cols)} 個):")
        for col in new_cols:
            print(f"   • {col}")


if __name__ == "__main__":
    process_all_stocks()
