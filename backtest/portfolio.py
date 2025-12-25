# -*- coding: utf-8 -*-
"""
投資組合回測引擎
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from .strategy_portfolio import PortfolioStrategy
from .metrics import calculate_metrics

class PortfolioEngine:
    """
    多股投資組合回測引擎
    """
    def __init__(self,
                 initial_capital: float = 1_000_000,
                 commission: float = 0.001425,
                 tax: float = 0.003,
                 slippage: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.tax = tax
        self.slippage = slippage
        
    def run(self, data_map: Dict[str, pd.DataFrame], strategy: PortfolioStrategy, 
            rebalance_freq='monthly') -> dict:
        """
        執行回測
        
        Args:
            data_map: {ticker: DataFrame} 股票資料字典，DF 需由舊到新排序
            strategy: 投資組合策略
            rebalance_freq: 再平衡頻率 (daily, weekly, monthly) - 尚未實作，全部依策略訊號
            
        Returns:
            dict: 回測結果
        """
        # 1. 時間軸對齊
        # 收集所有日期
        all_dates = set()
        ticker_data_lookup = {}  # 加速查詢 {ticker: {date_str: row}}
        
        print("⏳ 正在整理多股資料與時間軸...")
        
        # 預處理：轉成字典格式加速讀取，並找出所有日期
        for ticker, df in data_map.items():
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            # 確保有 date 欄位且為字串
            if 'date' not in df.columns:
                df['date'] = df.index.astype(str)
            else:
                df['date'] = df['date'].astype(str)
                
            # 建立日期索引的資料表
            df_dict = df.set_index('date').to_dict('index')
            ticker_data_lookup[ticker] = df_dict
            all_dates.update(df_dict.keys())
            
        # 排序日期
        sorted_dates = sorted(list(all_dates))
        print(f"✅ 時間軸建立完成: 共 {len(sorted_dates)} 個交易日")
        
        # 2. 初始化帳戶狀態
        cash = self.initial_capital
        # positions = {ticker: {'shares': 0, 'avg_cost': 0.0}}
        positions = {ticker: {'shares': 0, 'avg_cost': 0.0} for ticker in data_map.keys()}
        portfolio_value_curve = []  # 總權益曲線
        trades = []
        
        # 3. 逐日模擬
        for current_date in sorted_dates:
            # 取得當日各股價格
            current_prices = {}
            data_slice = {}
            portfolio_market_value = 0
            
            # 先掃描今日有資料的股票
            for ticker in data_map.keys():
                row = ticker_data_lookup[ticker].get(current_date)
                if row:
                    price = row['close']
                    current_prices[ticker] = price
                    data_slice[ticker] = row
                    portfolio_market_value += positions[ticker]['shares'] * price
                elif positions[ticker]['shares'] > 0:
                    # 今日無資料但有持倉，使用平均成本估算市值 (或上日收盤)
                    # 為求精確應使用 Forward Fill，這裡簡化處理
                    # 假設價格不變
                    pass
                    # 若要更精確，應維持上一個已知價格，這裡暫略

            # 計算當日總權益 (NAV)
            total_equity = cash + portfolio_market_value
            portfolio_value_curve.append({
                'date': current_date,
                'equity': total_equity,
                'cash': cash,
                'market_value': portfolio_market_value
            })
            
            # --- 再平衡邏輯 ---
            available_tickers = list(current_prices.keys())
            if not available_tickers:
                continue
                
            target_weights = strategy.rebalance_signal(current_date, available_tickers, data_slice, positions)
            
            if target_weights is not None:
                # 檢查是否為 DCA 模式（純買入，不賣出）
                is_dca_mode = target_weights.pop('_dca_mode', False)
                monthly_amount = target_weights.pop('_monthly_amount', 0)
                is_first_buy = target_weights.pop('_is_first_buy', False)
                
                if is_dca_mode:
                    # DCA 模式：使用固定金額買入，不賣出
                    if is_first_buy:
                        # 第一次購買使用全部初始資金
                        buy_budget = cash
                    else:
                        # 之後每月「注入」新資金（模擬投資者每月定期存入）
                        # 這模擬真實的定期定額：每月自動從薪水帳戶轉入固定金額
                        cash += monthly_amount  # 注入新資金
                        buy_budget = monthly_amount
                    
                    if buy_budget <= 0:
                        continue
                    
                    # 等權重分配購買金額
                    per_ticker_budget = buy_budget / len(available_tickers)
                    
                    for ticker in available_tickers:
                        if ticker.startswith('_'):
                            continue
                        current_price = current_prices[ticker]
                        buy_price = current_price * (1 + self.slippage)
                        cost_factor = 1 + self.commission + self.slippage
                        
                        # 計算可購買股數
                        buy_shares = int(per_ticker_budget / (buy_price * cost_factor))
                        
                        if buy_shares > 0:
                            cost = buy_shares * buy_price
                            fee_comm = cost * self.commission
                            total_cost = cost + fee_comm
                            
                            if cash >= total_cost:
                                cash -= total_cost
                                
                                # 更新平均成本
                                old_shares = positions[ticker]['shares']
                                old_cost = positions[ticker]['avg_cost']
                                new_shares = old_shares + buy_shares
                                new_avg_cost = ((old_shares * old_cost) + total_cost) / new_shares
                                
                                positions[ticker]['shares'] = new_shares
                                positions[ticker]['avg_cost'] = new_avg_cost
                                
                                trades.append({
                                    'date': current_date,
                                    'ticker': ticker,
                                    'type': 'BUY',
                                    'shares': buy_shares,
                                    'price': buy_price,
                                    'amount': total_cost,
                                    'profit': 0,
                                    'reason': '定期定額買入' if not is_first_buy else '初始資金買入'
                                })
                else:
                    # 一般再平衡模式（包含賣出）
                    # 賣出邏輯 (先賣再買，釋放資金)
                    for ticker in available_tickers:
                        if ticker.startswith('_'):
                            continue
                        current_shares = positions[ticker]['shares']
                        current_price = current_prices[ticker]
                        
                        # 目標權重 (若無則為 0)
                        w = target_weights.get(ticker, 0.0)
                        target_value = total_equity * w
                        target_shares = int(target_value / current_price)
                        
                        diff_shares = target_shares - current_shares
                        
                        if diff_shares < 0:
                            # SELL
                            sell_shares = abs(diff_shares)
                            sell_price = current_price * (1 - self.slippage)
                            revenue = sell_shares * sell_price
                            fee_comm = revenue * self.commission
                            fee_tax = revenue * self.tax
                            net_revenue = revenue - fee_comm - fee_tax
                            
                            # 計算損益
                            avg_cost = positions[ticker]['avg_cost']
                            cost_basis = sell_shares * avg_cost
                            realized_pnl = net_revenue - cost_basis
                            
                            cash += net_revenue
                            positions[ticker]['shares'] -= sell_shares
                            if positions[ticker]['shares'] == 0:
                                positions[ticker]['avg_cost'] = 0.0
                            
                            trades.append({
                                'date': current_date,
                                'ticker': ticker,
                                'type': 'SELL',
                                'shares': sell_shares,
                                'price': sell_price,
                                'amount': net_revenue,
                                'profit': realized_pnl,
                                'reason': '再平衡賣出'
                            })
                    
                    # 買入邏輯
                    for ticker in available_tickers:
                        if ticker.startswith('_'):
                            continue
                        current_shares = positions[ticker]['shares']
                        current_price = current_prices[ticker]
                        w = target_weights.get(ticker, 0.0)
                        target_value = total_equity * w
                        
                        # 買入時要考慮成本
                        cost_factor = 1 + self.commission + self.slippage
                        target_shares = int(target_value / (current_price * cost_factor))
                        
                        diff_shares = target_shares - current_shares
                        
                        if diff_shares > 0:
                            # BUY
                            buy_shares = diff_shares
                            buy_price = current_price * (1 + self.slippage)
                            cost = buy_shares * buy_price
                            fee_comm = cost * self.commission
                            
                            total_cost = cost + fee_comm
                            
                            if cash >= total_cost:
                                cash -= total_cost
                                
                                # 更新平均成本
                                old_shares = positions[ticker]['shares']
                                old_cost = positions[ticker]['avg_cost']
                                new_shares = old_shares + buy_shares
                                new_avg_cost = ((old_shares * old_cost) + total_cost) / new_shares
                                
                                positions[ticker]['shares'] = new_shares
                                positions[ticker]['avg_cost'] = new_avg_cost
                                
                                trades.append({
                                    'date': current_date,
                                    'ticker': ticker,
                                    'type': 'BUY',
                                    'shares': buy_shares,
                                    'price': buy_price,
                                    'amount': total_cost,
                                    'profit': 0,
                                    'reason': '再平衡買入'
                                })
        
        # 4. 結算 - 計算最終市值
        final_market_value = 0
        final_prices = {}
        for ticker, pos in positions.items():
            if pos['shares'] > 0:
                # 嘗試取得最後一天的價格
                last_row = ticker_data_lookup[ticker].get(sorted_dates[-1])
                if last_row:
                    price = last_row['close']
                    final_prices[ticker] = price
                    final_market_value += pos['shares'] * price
        
        # 總權益 = 現金 + 持股市值
        final_equity = cash + final_market_value
        
        df_result = pd.DataFrame(portfolio_value_curve)
        df_result['date'] = pd.to_datetime(df_result['date'])
        df_result.set_index('date', inplace=True)
        
        # 計算績效指標 (重複利用 metrics 模組)
        equity_series = df_result['equity']
        metrics = calculate_metrics(trades, equity_series, self.initial_capital)
        
        # 覆蓋 final_capital 為正確的總權益
        metrics['final_capital'] = round(final_equity, 2)
        metrics['final_equity'] = round(final_equity, 2)  # 報告用這個 key
        metrics['cash'] = round(cash, 2)
        metrics['market_value'] = round(final_market_value, 2)
        metrics['total_trades'] = len(trades)  # 報告用這個 key
        
        # 轉換 positions 格式回傳 (僅回傳 shares 方便閱讀)
        final_positions = {k: v['shares'] for k, v in positions.items() if v['shares'] > 0}
        
        return {
            'metrics': metrics,
            'equity_curve': equity_series,
            'trades': pd.DataFrame(trades),
            'positions': final_positions,
            'final_prices': final_prices
        }
