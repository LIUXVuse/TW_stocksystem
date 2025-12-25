# -*- coding: utf-8 -*-
"""
批次回測模組
支援多股票批次回測和全市場掃描
"""
import os
import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 確保可以導入同目錄模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .engine import BacktestEngine
from .strategy import Strategy


# 資料目錄
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK_DIR = os.path.join(BASE_DIR, "data", "tw-share", "dayK")


def get_all_stock_files() -> list:
    """取得所有股票 CSV 檔案"""
    return glob(os.path.join(STOCK_DIR, "*.csv"))


def extract_ticker_from_path(path: str) -> str:
    """從檔案路徑提取股票代碼"""
    basename = os.path.basename(path)
    return basename.split('_')[0]


def batch_backtest(tickers: list = None,
                   strategy: Strategy = None,
                   initial_capital: float = 1_000_000,
                   top_n: int = 20,
                   sort_by: str = 'total_return',
                   ascending: bool = False,
                   min_trades: int = 3,
                   show_progress: bool = True) -> pd.DataFrame:
    """
    批次回測多支股票
    
    Args:
        tickers: 股票代碼列表（如 ['2330.TW', '2317.TW']），None 則測試全部
        strategy: 策略物件
        initial_capital: 初始資金
        top_n: 回傳前 N 名結果（0 = 全部）
        sort_by: 排序依據欄位
        ascending: 是否升序排列
        min_trades: 最少交易次數過濾
        show_progress: 是否顯示進度條
    
    Returns:
        pd.DataFrame: 按績效排序的結果表
    """
    if strategy is None:
        raise ValueError("必須提供 strategy 參數")
    
    # 取得要測試的檔案
    if tickers is None:
        files = get_all_stock_files()
    else:
        files = []
        all_files = get_all_stock_files()
        for t in tickers:
            matching = [f for f in all_files if t in os.path.basename(f)]
            files.extend(matching)
    
    if not files:
        raise ValueError("找不到符合條件的股票檔案")
    
    engine = BacktestEngine(initial_capital=initial_capital)
    results = []
    
    iterator = tqdm(files, desc="批次回測") if show_progress else files
    
    for csv_path in iterator:
        ticker = extract_ticker_from_path(csv_path)
        
        try:
            df = pd.read_csv(csv_path)
            result = engine.run(df, strategy, verbose=False)
            
            metrics = result['metrics'].copy()
            metrics['ticker'] = ticker
            metrics['name'] = os.path.basename(csv_path).replace('.csv', '').split('_', 1)[-1]
            
            results.append(metrics)
            
        except Exception as e:
            continue
    
    if not results:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    
    # 過濾最少交易次數
    if min_trades > 0:
        df_results = df_results[df_results['trade_count'] >= min_trades]
    
    # 排序
    df_results = df_results.sort_values(sort_by, ascending=ascending)
    
    # 限制結果數量
    if top_n > 0:
        df_results = df_results.head(top_n)
    
    # 重新排列欄位
    priority_cols = ['ticker', 'name', 'total_return', 'annual_return', 
                     'sharpe_ratio', 'max_drawdown', 'win_rate', 'trade_count']
    other_cols = [c for c in df_results.columns if c not in priority_cols]
    df_results = df_results[priority_cols + other_cols]
    
    return df_results.reset_index(drop=True)


def market_scan(strategy: Strategy,
                filter_func: callable = None,
                top_n: int = 30,
                sort_by: str = 'sharpe_ratio') -> pd.DataFrame:
    """
    全市場掃描
    
    Args:
        strategy: 策略物件
        filter_func: 過濾函數 (df) -> bool
        top_n: 回傳前 N 名
        sort_by: 排序依據
    
    Returns:
        pd.DataFrame: 績效最佳股票列表
    """
    files = get_all_stock_files()
    engine = BacktestEngine()
    results = []
    
    for csv_path in tqdm(files, desc="掃描市場"):
        try:
            df = pd.read_csv(csv_path)
            
            # 應用過濾條件
            if filter_func and not filter_func(df):
                continue
            
            result = engine.run(df, strategy, verbose=False)
            
            # 只記錄有交易的結果
            if result['metrics']['trade_count'] > 0:
                metrics = result['metrics'].copy()
                metrics['ticker'] = extract_ticker_from_path(csv_path)
                results.append(metrics)
                
        except:
            continue
    
    if not results:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(sort_by, ascending=False)
    
    return df_results.head(top_n).reset_index(drop=True)


def compare_strategies(tickers: list,
                       strategies: list,
                       aggregate: str = 'mean') -> pd.DataFrame:
    """
    比較多個策略在多支股票上的表現
    
    Args:
        tickers: 股票代碼列表
        strategies: 策略物件列表
        aggregate: 聚合方式 ('mean', 'median', 'sum')
    
    Returns:
        pd.DataFrame: 策略比較表
    """
    engine = BacktestEngine()
    strategy_results = {s.name: [] for s in strategies}
    
    files = get_all_stock_files()
    
    for ticker in tqdm(tickers, desc="比較策略"):
        matching = [f for f in files if ticker in os.path.basename(f)]
        if not matching:
            continue
            
        try:
            df = pd.read_csv(matching[0])
            
            for strategy in strategies:
                result = engine.run(df, strategy, verbose=False)
                strategy_results[strategy.name].append(result['metrics'])
                
        except:
            continue
    
    # 聚合結果
    summary = []
    for name, results_list in strategy_results.items():
        if not results_list:
            continue
            
        df_s = pd.DataFrame(results_list)
        
        if aggregate == 'mean':
            agg = df_s.mean(numeric_only=True)
        elif aggregate == 'median':
            agg = df_s.median(numeric_only=True)
        else:
            agg = df_s.sum(numeric_only=True)
        
        agg['strategy'] = name
        agg['stocks_tested'] = len(results_list)
        summary.append(agg)
    
    return pd.DataFrame(summary)


if __name__ == '__main__':
    from strategy import MACrossStrategy, RSIStrategy
    
    print("📊 測試批次回測...")
    
    # 測試熱門股票
    tickers = ['2330.TW', '2317.TW', '2454.TW', '0050.TW', '0056.TW']
    strategy = MACrossStrategy(5, 20)
    
    results = batch_backtest(
        tickers=tickers,
        strategy=strategy,
        top_n=10
    )
    
    print("\n📈 回測結果:")
    print(results[['ticker', 'name', 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']].to_string())
