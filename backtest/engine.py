# -*- coding: utf-8 -*-
"""
回測引擎核心
"""
import pandas as pd
import numpy as np
from .strategy import Strategy
from .metrics import calculate_metrics, print_metrics


class BacktestEngine:
    """
    回測引擎
    
    模擬交易過程並計算績效
    """
    
    def __init__(self,
                 initial_capital: float = 1_000_000,
                 commission: float = 0.001425,    # 手續費 0.1425%
                 tax: float = 0.003,              # 證交稅 0.3% (賣出時收)
                 slippage: float = 0.001):        # 滑價 0.1%
        """
        初始化回測引擎
        
        Args:
            initial_capital: 初始資金（預設 100 萬）
            commission: 手續費率（預設 0.1425%）
            tax: 證交稅率（預設 0.3%）
            slippage: 滑價率（預設 0.1%）
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.tax = tax
        self.slippage = slippage
    
    def run(self, df: pd.DataFrame, strategy: Strategy, 
            position_size: float = 1.0,
            verbose: bool = False) -> dict:
        """
        執行回測
        
        Args:
            df: 包含 OHLCV 和技術指標的 DataFrame
            strategy: 策略物件
            position_size: 持倉比例（0-1，預設全倉）
            verbose: 是否印出詳細資訊
        
        Returns:
            dict: {
                'trades': 交易明細,
                'equity_curve': 權益曲線,
                'metrics': 績效指標,
                'signals': 訊號序列
            }
        """
        # 複製資料避免修改原始 DataFrame
        df = df.copy()
        
        # 確保欄位名稱為小寫
        df.columns = [c.lower() for c in df.columns]
        
        # 產生訊號
        signals = strategy.generate_signals(df)
        
        # 初始化
        capital = self.initial_capital
        position = 0  # 持股數量
        entry_price = 0  # 進場價格
        trades = []  # 交易記錄
        equity_curve = []  # 權益曲線
        
        for i, (idx, row) in enumerate(df.iterrows()):
            signal = signals.iloc[i]
            price = row['close']
            
            # 計算目前權益
            current_equity = capital + position * price
            equity_curve.append(current_equity)
            
            # 處理訊號
            if signal == 1 and position == 0:
                # 買入訊號且無持倉
                buy_price = price * (1 + self.slippage)  # 滑價
                shares = int((capital * position_size) / buy_price)
                
                if shares > 0:
                    cost = shares * buy_price
                    commission_fee = cost * self.commission
                    
                    position = shares
                    entry_price = buy_price
                    capital -= (cost + commission_fee)
                    
                    trade = {
                        'type': 'BUY',
                        'date': str(row['date'])[:10] if 'date' in row else str(idx)[:10],
                        'price': round(buy_price, 2),
                        'shares': shares,
                        'cost': round(cost + commission_fee, 2)
                    }
                    trades.append(trade)
                    
                    if verbose:
                        print(f"BUY: {trade['date']} @ ${trade['price']:.2f} x {shares}")
            
            elif signal == -1 and position > 0:
                # 賣出訊號且有持倉
                sell_price = price * (1 - self.slippage)  # 滑價
                revenue = position * sell_price
                commission_fee = revenue * self.commission
                tax_fee = revenue * self.tax
                
                net_revenue = revenue - commission_fee - tax_fee
                profit = net_revenue - (entry_price * position)
                
                trade = {
                    'type': 'SELL',
                    'date': str(row['date'])[:10] if 'date' in row else str(idx)[:10],
                    'price': round(sell_price, 2),
                    'shares': position,
                    'revenue': round(net_revenue, 2),
                    'profit': round(profit, 2),
                    'return': round(profit / (entry_price * position), 4)
                }
                trades.append(trade)
                
                if verbose:
                    print(f"SELL: {trade['date']} @ ${trade['price']:.2f}, "
                          f"profit: ${trade['profit']:,.0f} ({trade['return']:.2%})")
                
                capital += net_revenue
                position = 0
                entry_price = 0
        
        # 如果結束時還有持倉，以最後價格計算
        if position > 0:
            final_price = df['close'].iloc[-1]
            final_equity = capital + position * final_price
        else:
            final_equity = capital
        
        equity_curve[-1] = final_equity
        
        # 轉換為 Series
        equity_series = pd.Series(equity_curve, index=df.index)
        
        # 計算績效指標
        metrics = calculate_metrics(trades, equity_series, self.initial_capital)
        metrics['strategy'] = strategy.name
        
        return {
            'trades': trades,
            'equity_curve': equity_series,
            'metrics': metrics,
            'signals': signals
        }
    
    def run_multiple(self, df: pd.DataFrame, strategies: list,
                     verbose: bool = False) -> pd.DataFrame:
        """
        執行多策略回測比較
        
        Args:
            df: 資料 DataFrame
            strategies: 策略物件列表
            verbose: 是否印出詳細資訊
        
        Returns:
            pd.DataFrame: 各策略績效比較表
        """
        results = []
        
        for strategy in strategies:
            result = self.run(df, strategy, verbose=verbose)
            metrics = result['metrics']
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def optimize(self, df: pd.DataFrame, strategy_class: type,
                 param_grid: dict, metric: str = 'sharpe_ratio',
                 verbose: bool = False) -> dict:
        """
        參數優化
        
        Args:
            df: 股價資料
            strategy_class: 策略類別（非實例）
            param_grid: 參數網格 {'short_period': [5,10,20], 'long_period': [20,60]}
            metric: 優化目標指標 (sharpe_ratio, total_return, max_drawdown, win_rate)
            verbose: 是否印出詳細資訊
        
        Returns:
            dict: {
                'best_params': 最佳參數,
                'best_score': 最佳分數,
                'best_result': 最佳回測結果,
                'all_results': 所有測試結果 DataFrame
            }
        """
        from itertools import product
        
        # 產生所有參數組合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        if verbose:
            print(f"📊 參數優化: 測試 {len(combinations)} 種組合")
        
        all_results = []
        best_score = float('-inf') if metric != 'max_drawdown' else float('inf')
        best_params = None
        best_result = None
        
        for combo in combinations:
            # 建立參數字典
            params = dict(zip(param_names, combo))
            
            try:
                # 建立策略實例
                strategy = strategy_class(**params)
                
                # 執行回測
                result = self.run(df, strategy, verbose=False)
                
                # 記錄結果
                record = params.copy()
                record.update(result['metrics'])
                all_results.append(record)
                
                # 檢查是否為最佳
                score = result['metrics'].get(metric, 0)
                
                if metric == 'max_drawdown':
                    # 回撤越小越好（越接近 0）
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_result = result
                else:
                    # 其他指標越大越好
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_result = result
                        
            except Exception as e:
                if verbose:
                    print(f"  ❌ 參數 {params} 失敗: {e}")
                continue
        
        if verbose and best_params:
            print(f"\n🏆 最佳參數: {best_params}")
            print(f"   {metric}: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_result': best_result,
            'all_results': pd.DataFrame(all_results)
        }


def quick_backtest(csv_path: str, strategy: Strategy, 
                   initial_capital: float = 1_000_000,
                   show_report: bool = True) -> dict:
    """
    快速回測工具函數
    
    Args:
        csv_path: CSV 檔案路徑
        strategy: 策略物件
        initial_capital: 初始資金
        show_report: 是否印出報告
    
    Returns:
        dict: 回測結果
    """
    df = pd.read_csv(csv_path)
    engine = BacktestEngine(initial_capital=initial_capital)
    result = engine.run(df, strategy)
    
    if show_report:
        print_metrics(result['metrics'])
    
    return result
