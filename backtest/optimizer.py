# -*- coding: utf-8 -*-
"""
策略參數優化器
透過 Grid Search 找出最佳策略參數
"""
import os
import sys
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# 確保可以匯入專案模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine
from backtest.strategy import MACrossStrategy, MACDStrategy, RSIStrategy


class StrategyOptimizer:
    """
    策略參數優化器
    """
    def __init__(self, initial_capital: float = 1_000_000, min_trades: int = 3):
        self.initial_capital = initial_capital
        self.min_trades = min_trades  # 最低交易次數過濾
        self.engine = BacktestEngine(initial_capital=initial_capital)
    
    def _ensure_ma_columns(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        確保 DataFrame 有需要的 MA 欄位，沒有就自動計算
        """
        df = df.copy()
        for period in periods:
            col_name = f'ma{period}'
            if col_name not in df.columns:
                df[col_name] = df['close'].rolling(window=period).mean()
        return df
        
    def grid_search(self, 
                    df: pd.DataFrame, 
                    strategy_class, 
                    param_grid: Dict[str, List[Any]],
                    metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        Grid Search 參數優化
        
        Args:
            df: 股價 DataFrame
            strategy_class: 策略類別 (e.g. MACrossStrategy)
            param_grid: 參數網格 e.g. {'short_period': [5, 10], 'long_period': [20, 40]}
            metric: 優化目標指標 ('sharpe_ratio', 'total_return', 'max_drawdown')
            
        Returns:
            pd.DataFrame: 所有組合的回測結果
        """
        # 產生所有參數組合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        # 過濾無效組合 (short >= long)
        valid_combinations = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            if 'short_period' in params and 'long_period' in params:
                if params['short_period'] >= params['long_period']:
                    continue
            valid_combinations.append(combo)
        
        print(f"🔍 開始參數優化: {len(valid_combinations)} 種有效組合")
        print(f"   策略: {strategy_class.__name__}")
        print(f"   優化目標: {metric}")
        print(f"   最低交易次數: {self.min_trades}")
        print("-" * 50)
        
        # 預先計算所有需要的 MA 欄位
        all_periods = set()
        for name, values in param_grid.items():
            if 'period' in name.lower():
                all_periods.update(values)
        
        if all_periods:
            print(f"   📊 自動計算 MA 欄位: {sorted(all_periods)}")
            df = self._ensure_ma_columns(df, list(all_periods))
        
        results = []
        
        for i, combo in enumerate(valid_combinations, 1):
            # 建立參數字典
            params = dict(zip(param_names, combo))
            
            try:
                # 建立策略
                strategy = strategy_class(**params)
                
                # 執行回測
                result = self.engine.run(df, strategy)
                metrics = result.get('metrics', {})
                
                # 記錄結果
                record = {**params}
                record['sharpe_ratio'] = metrics.get('sharpe_ratio', 0)
                record['total_return'] = metrics.get('total_return', 0)
                record['annual_return'] = metrics.get('annual_return', 0)
                record['max_drawdown'] = metrics.get('max_drawdown', 0)
                record['win_rate'] = metrics.get('win_rate', 0)
                record['profit_factor'] = metrics.get('profit_factor', 0)
                record['total_trades'] = metrics.get('trade_count', 0)
                
                results.append(record)
                
                # 顯示進度
                if i % 5 == 0 or i == len(valid_combinations):
                    print(f"   進度: {i}/{len(valid_combinations)} ({i/len(valid_combinations)*100:.0f}%)")
                    
            except Exception as e:
                print(f"   ⚠️ 參數組合 {params} 失敗: {e}")
                continue
        
        # 轉成 DataFrame 並排序
        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            # 過濾交易次數過少的組合
            df_filtered = df_results[df_results['total_trades'] >= self.min_trades]
            
            if not df_filtered.empty:
                # 根據目標指標排序
                ascending = True if metric == 'max_drawdown' else False
                df_results = df_filtered.sort_values(metric, ascending=ascending).reset_index(drop=True)
                print(f"   📌 過濾後剩餘 {len(df_results)} 個組合 (交易次數 >= {self.min_trades})")
            else:
                print(f"   ⚠️ 沒有組合達到最低交易次數 {self.min_trades}，顯示所有結果")
                ascending = True if metric == 'max_drawdown' else False
                df_results = df_results.sort_values(metric, ascending=ascending).reset_index(drop=True)
            
        print("-" * 50)
        print(f"✅ 優化完成！找到 {len(df_results)} 個有效組合")
        
        return df_results
    
    def generate_optimization_report(self, 
                                      df_results: pd.DataFrame, 
                                      strategy_name: str,
                                      ticker: str,
                                      save_path: str = None) -> str:
        """
        產生參數優化報告 HTML
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # 最佳參數 (第一行)
        if not df_results.empty:
            best = df_results.iloc[0]
            best_params = {k: v for k, v in best.items() if k not in ['sharpe_ratio', 'total_return', 'annual_return', 'max_drawdown', 'win_rate', 'profit_factor', 'total_trades']}
        else:
            best = {}
            best_params = {}
        
        # 表格行
        table_rows = []
        for i, row in df_results.head(20).iterrows():
            param_cols = [k for k in row.index if k not in ['sharpe_ratio', 'total_return', 'annual_return', 'max_drawdown', 'win_rate', 'profit_factor', 'total_trades']]
            param_str = ', '.join([f"{k}={row[k]}" for k in param_cols])
            return_class = 'positive' if row['total_return'] > 0 else 'negative'
            
            table_rows.append(f'''
            <tr>
                <td>#{i+1}</td>
                <td>{param_str}</td>
                <td>{row['sharpe_ratio']:.2f}</td>
                <td class="{return_class}">{row['total_return']:.2%}</td>
                <td>{row['max_drawdown']:.2%}</td>
                <td>{row['win_rate']:.2%}</td>
                <td>{int(row['total_trades'])}</td>
            </tr>''')
        
        table_html = '\n'.join(table_rows)
        
        html = f'''
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>參數優化報告 - {strategy_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #f0883e, #8957e5); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
        .card {{ background: #161b22; border-radius: 10px; padding: 20px; margin-bottom: 20px; border: 1px solid #30363d; }}
        .card h2 {{ color: #58a6ff; margin-bottom: 15px; }}
        .best-params {{ display: flex; gap: 15px; flex-wrap: wrap; }}
        .param-badge {{ background: #238636; padding: 10px 20px; border-radius: 20px; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #30363d; }}
        th {{ background: #21262d; color: #58a6ff; }}
        tr:hover {{ background: #1f2937; }}
        .positive {{ color: #3fb950; }}
        .negative {{ color: #f85149; }}
        .footer {{ text-align: center; color: #8b949e; margin-top: 30px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 參數優化報告</h1>
            <p>策略: {strategy_name} | 股票: {ticker} | 產生時間: {now}</p>
        </div>
        
        <div class="card">
            <h2>🏆 最佳參數組合</h2>
            <div class="best-params">
                {' '.join([f'<span class="param-badge">{k} = {v}</span>' for k, v in best_params.items()])}
            </div>
            <p style="margin-top: 15px; color: #8b949e;">夏普比率: {best.get('sharpe_ratio', 0):.2f} | 報酬率: {best.get('total_return', 0):.2%}</p>
        </div>
        
        <div class="card">
            <h2>📊 所有組合排名 (Top 20)</h2>
            <table>
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>參數</th>
                        <th>夏普</th>
                        <th>報酬率</th>
                        <th>最大回撤</th>
                        <th>勝率</th>
                        <th>交易次數</th>
                    </tr>
                </thead>
                <tbody>
                    {table_html}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            由 tw-quant-simple 參數優化器產生
        </div>
    </div>
</body>
</html>
'''
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"✅ 優化報告已儲存: {save_path}")
            
        return html


def run_optimization_example():
    """
    執行範例優化
    """
    from glob import glob
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'tw-share', 'dayK', '2330.TW*.csv')
    files = glob(data_path)
    
    if not files:
        print("❌ 找不到 2330.TW 的資料")
        return
        
    df = pd.read_csv(files[0])
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'date'}, inplace=True)
    df = df.sort_values('date').reset_index(drop=True)
    
    # 優化器
    optimizer = StrategyOptimizer()
    
    # MA 交叉策略參數網格
    param_grid = {
        'short_period': [5, 10, 15, 20],
        'long_period': [20, 40, 60, 120]
    }
    
    # 執行優化
    results = optimizer.grid_search(df, MACrossStrategy, param_grid, metric='sharpe_ratio')
    
    # 顯示結果
    print("\n📋 優化結果 (Top 10):")
    print(results.head(10).to_string(index=False))
    
    # 產生報告
    report_path = os.path.join(base_dir, 'reports', 'optimization_report.html')
    optimizer.generate_optimization_report(results, 'MA Cross', '2330.TW', save_path=report_path)
    
    # 匯出 CSV
    csv_path = os.path.join(base_dir, 'reports', 'optimization_results.csv')
    results.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 優化結果已匯出: {csv_path}")


if __name__ == '__main__':
    run_optimization_example()
