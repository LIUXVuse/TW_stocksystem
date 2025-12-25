# -*- coding: utf-8 -*-
"""
績效指標計算
"""
import pandas as pd
import numpy as np


def calculate_metrics(trades: list, equity_curve: pd.Series, 
                      initial_capital: float, risk_free_rate: float = 0.02) -> dict:
    """
    計算回測績效指標
    
    Args:
        trades: 交易記錄列表
        equity_curve: 權益曲線
        initial_capital: 初始資金
        risk_free_rate: 無風險利率（年化，預設 2%）
    
    Returns:
        dict: 績效指標字典
    """
    metrics = {}
    
    # ========== 報酬率指標 ==========
    final_capital = equity_curve.iloc[-1]
    total_return = (final_capital - initial_capital) / initial_capital
    metrics['initial_capital'] = initial_capital
    metrics['final_capital'] = round(final_capital, 2)
    metrics['total_return'] = round(total_return, 4)
    
    # 年化報酬率
    trading_days = len(equity_curve)
    if trading_days > 0:
        annual_return = (1 + total_return) ** (252 / trading_days) - 1
        metrics['annual_return'] = round(annual_return, 4)
    else:
        metrics['annual_return'] = 0
    
    # ========== 風險指標 ==========
    
    # 每日報酬率
    daily_returns = equity_curve.pct_change().dropna()
    
    # 波動率（年化）
    volatility = daily_returns.std() * np.sqrt(252)
    metrics['volatility'] = round(volatility, 4) if not np.isnan(volatility) else 0
    
    # 夏普比率
    if metrics['volatility'] > 0:
        sharpe_ratio = (metrics['annual_return'] - risk_free_rate) / metrics['volatility']
        metrics['sharpe_ratio'] = round(sharpe_ratio, 2)
    else:
        metrics['sharpe_ratio'] = 0
    
    # 最大回撤
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    metrics['max_drawdown'] = round(max_drawdown, 4) if not np.isnan(max_drawdown) else 0
    
    # 最大回撤期間
    drawdown_start = drawdown.idxmin() if not drawdown.empty else None
    metrics['max_drawdown_date'] = str(drawdown_start)[:10] if drawdown_start else "N/A"
    
    # ========== 交易統計 ==========
    
    if trades:
        # 計算盈虧
        profits = [t['profit'] for t in trades if 'profit' in t]
        
        metrics['trade_count'] = len(trades)
        
        if profits:
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p < 0]
            
            # 勝率
            metrics['win_rate'] = round(len(wins) / len(profits), 4) if profits else 0
            
            # 平均獲利/虧損
            metrics['avg_win'] = round(np.mean(wins), 2) if wins else 0
            metrics['avg_loss'] = round(np.mean(losses), 2) if losses else 0
            
            # 盈虧比
            if metrics['avg_loss'] != 0:
                metrics['profit_factor'] = round(abs(metrics['avg_win'] / metrics['avg_loss']), 2)
            else:
                metrics['profit_factor'] = float('inf') if metrics['avg_win'] > 0 else 0
            
            # 總獲利/虧損
            metrics['total_profit'] = round(sum(wins), 2) if wins else 0
            metrics['total_loss'] = round(sum(losses), 2) if losses else 0
        else:
            metrics['win_rate'] = 0
            metrics['avg_win'] = 0
            metrics['avg_loss'] = 0
            metrics['profit_factor'] = 0
            metrics['total_profit'] = 0
            metrics['total_loss'] = 0
    else:
        metrics['trade_count'] = 0
        metrics['win_rate'] = 0
        metrics['avg_win'] = 0
        metrics['avg_loss'] = 0
        metrics['profit_factor'] = 0
        metrics['total_profit'] = 0
        metrics['total_loss'] = 0
    
    return metrics


def print_metrics(metrics: dict):
    """
    格式化印出績效指標
    """
    print("=" * 50)
    print("📊 回測績效報告")
    print("=" * 50)
    
    print("\n📈 報酬率")
    print(f"   初始資金: ${metrics['initial_capital']:,.0f}")
    print(f"   最終資金: ${metrics['final_capital']:,.0f}")
    print(f"   總報酬率: {metrics['total_return']:.2%}")
    print(f"   年化報酬: {metrics['annual_return']:.2%}")
    
    print("\n📉 風險指標")
    print(f"   年化波動: {metrics['volatility']:.2%}")
    print(f"   夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"   最大回撤: {metrics['max_drawdown']:.2%}")
    
    print("\n🔄 交易統計")
    print(f"   交易次數: {metrics['trade_count']} 筆")
    print(f"   勝率: {metrics['win_rate']:.2%}")
    print(f"   盈虧比: {metrics['profit_factor']:.2f}")
    print(f"   平均獲利: ${metrics['avg_win']:,.0f}")
    print(f"   平均虧損: ${metrics['avg_loss']:,.0f}")
    
    print("=" * 50)
