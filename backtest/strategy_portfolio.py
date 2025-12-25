# -*- coding: utf-8 -*-
"""
投資組合策略模組 (重構版)
支援多種再平衡模式和個股獨立策略
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


class PortfolioStrategy:
    """
    投資組合策略基類
    """
    def __init__(self, name="Portfolio Strategy"):
        self.name = name
        
    def rebalance_signal(self, current_date, available_tickers: list, 
                         data_slice: dict, positions: dict = None) -> dict:
        """
        產生再平衡訊號
        
        Returns:
            dict: 目標權重 {ticker: target_weight}
                  None 表示不調整
        """
        raise NotImplementedError


class EqualWeightMonthlyStrategy(PortfolioStrategy):
    """
    等權重再平衡策略
    
    支援三種頻率：
    - weekly: 每週一再平衡
    - monthly: 每月第一個交易日再平衡
    - quarterly: 每季第一個交易日再平衡
    """
    def __init__(self, top_n=None, freq="monthly"):
        super().__init__(name="Equal Weight")
        self.top_n = top_n
        self.freq = freq  # weekly, monthly, quarterly
        self.last_rebalance_period = None
        
    def _get_period_key(self, dt):
        """根據頻率取得週期 key"""
        if self.freq == "weekly":
            # ISO calendar: (year, week_number)
            iso = dt.isocalendar()
            return (iso[0], iso[1])
        elif self.freq == "quarterly":
            quarter = (dt.month - 1) // 3 + 1
            return (dt.year, quarter)
        else:  # monthly
            return (dt.year, dt.month)
    
    def _is_valid_rebalance_day(self, dt):
        """檢查是否為有效的再平衡日"""
        if self.freq == "weekly":
            # 週一 = 0
            return dt.weekday() == 0
        else:
            # 月初或季初：由上層判斷（每個新週期的第一個交易日）
            return True
        
    def rebalance_signal(self, current_date, available_tickers: list, 
                         data_slice: dict, positions: dict = None) -> dict:
        """根據設定頻率進行再平衡"""
        # 解析日期
        if isinstance(current_date, str):
            try:
                dt = datetime.strptime(current_date, '%Y-%m-%d')
            except:
                dt = datetime.strptime(current_date[:10], '%Y-%m-%d')
        else:
            dt = current_date
        
        current_period = self._get_period_key(dt)
        
        # 檢查是否需要再平衡（新的週期）
        if self.last_rebalance_period == current_period:
            return None  # 這個週期已經再平衡過了
        
        # 週頻：必須是週一
        if self.freq == "weekly" and not self._is_valid_rebalance_day(dt):
            return None
        
        self.last_rebalance_period = current_period
        
        # 執行再平衡
        candidates = available_tickers
        if self.top_n and len(candidates) > self.top_n:
            candidates = candidates[:self.top_n]
            
        if not candidates:
            return {}
            
        weight = 1.0 / len(candidates)
        return {ticker: weight for ticker in candidates}


# 別名，保持向後相容
EqualWeightStrategy = EqualWeightMonthlyStrategy


class BuyAndHoldStrategy(PortfolioStrategy):
    """
    買入持有策略
    
    支援三種模式：
    - diamond: 鑽石手（永不賣出）
    - rebuy: 停損停利後買回（達標賣出，冷靜期後買回）
    - multilayer: 多層次鑽石手（永不賣出 + 依時間表加碼）
    """
    def __init__(self, stop_loss: float = None, 
                 take_profit: float = None,
                 trailing_stop: float = None,
                 mode: str = "diamond",
                 cooldown_days: int = 30,
                 rebuy_amount: str = "all",
                 extra_buys: list = None):
        super().__init__(name="Buy & Hold")
        self.invested = False
        self.stop_loss = stop_loss  # 如 -0.10
        self.take_profit = take_profit  # 如 0.30
        self.trailing_stop = trailing_stop  # 如 -0.10
        self.mode = mode
        self.cooldown_days = cooldown_days
        self.rebuy_amount = rebuy_amount
        self.extra_buys = extra_buys or []
        
        # 追蹤最高價（用於移動停損）
        self.high_prices = {}
        self.entry_prices = {}
        self.initial_investment = 0  # 記錄初始投入金額
        
        # rebuy 模式的狀態
        self.sold_date = None  # 上次賣出日期
        self.waiting_rebuy = False  # 是否等待買回
        
        # multilayer 模式：已執行的加碼
        self.executed_extra_buys = set()
        
    def rebalance_signal(self, current_date, available_tickers: list, 
                         data_slice: dict, positions: dict = None) -> dict:
        positions = positions or {}
        
        # 解析日期
        if isinstance(current_date, str):
            try:
                dt = datetime.strptime(current_date, '%Y-%m-%d')
            except:
                dt = datetime.strptime(current_date[:10], '%Y-%m-%d')
        else:
            dt = current_date
        
        # ==== 鑽石手模式 ====
        if self.mode == "diamond":
            # 只買一次，永不賣出
            if not self.invested:
                self.invested = True
                weight = 1.0 / len(available_tickers) if available_tickers else 0
                return {ticker: weight for ticker in available_tickers}
            return None  # 永遠不賣
        
        # ==== 多層次鑽石手模式 ====
        elif self.mode == "multilayer":
            result = {}
            
            # 第一次買入
            if not self.invested:
                self.invested = True
                weight = 1.0 / len(available_tickers) if available_tickers else 0
                for ticker in available_tickers:
                    result[ticker] = weight
                    if ticker in data_slice:
                        self.entry_prices[ticker] = data_slice[ticker]['close']
                return result if result else None
            
            # 檢查是否有加碼計畫要執行
            current_date_str = dt.strftime('%Y-%m-%d')
            for extra in self.extra_buys:
                extra_date = extra.get('date', '')
                extra_amount = extra.get('amount', 0)
                
                # 如果今天 >= 加碼日期 且 尚未執行
                if extra_date and current_date_str >= extra_date and extra_date not in self.executed_extra_buys:
                    self.executed_extra_buys.add(extra_date)
                    # 使用特殊結構告訴引擎注入資金
                    return {
                        '_inject_cash': True,
                        '_inject_amount': extra_amount,
                        '_reason': f'加碼 ${extra_amount:,.0f}',
                        **{ticker: 1.0 / len(available_tickers) for ticker in available_tickers}
                    }
            
            return None  # 沒有加碼計畫
        
        # ==== 停損停利後買回模式 ====
        elif self.mode == "rebuy":
            # 等待買回狀態
            if self.waiting_rebuy:
                if self.sold_date:
                    days_since_sold = (dt - self.sold_date).days
                    if days_since_sold >= self.cooldown_days:
                        # 冷靜期已過，買回
                        self.waiting_rebuy = False
                        self.sold_date = None
                        weight = 1.0 / len(available_tickers) if available_tickers else 0
                        
                        # 記錄新的進場價
                        for ticker in available_tickers:
                            if ticker in data_slice:
                                self.entry_prices[ticker] = data_slice[ticker]['close']
                                self.high_prices[ticker] = data_slice[ticker]['close']
                        
                        return {
                            '_reason': f'冷靜期{self.cooldown_days}天後買回',
                            **{ticker: weight for ticker in available_tickers}
                        }
                return None  # 繼續等待
            
            # 第一次買入
            if not self.invested:
                self.invested = True
                weight = 1.0 / len(available_tickers) if available_tickers else 0
                
                for ticker in available_tickers:
                    if ticker in data_slice:
                        self.entry_prices[ticker] = data_slice[ticker]['close']
                        self.high_prices[ticker] = data_slice[ticker]['close']
                
                return {ticker: weight for ticker in available_tickers}
            
            # 檢查停損/停利條件
            sell_tickers = []
            for ticker in list(positions.keys()):
                if positions.get(ticker, {}).get('shares', 0) <= 0:
                    continue
                    
                if ticker not in data_slice:
                    continue
                    
                current_price = data_slice[ticker]['close']
                entry_price = self.entry_prices.get(ticker, current_price)
                high_price = self.high_prices.get(ticker, current_price)
                
                # 更新最高價
                if current_price > high_price:
                    self.high_prices[ticker] = current_price
                    high_price = current_price
                
                # 計算收益率
                pnl_pct = (current_price - entry_price) / entry_price
                
                # 檢查停損
                if self.stop_loss and pnl_pct <= self.stop_loss:
                    sell_tickers.append(ticker)
                    continue
                    
                # 檢查停利
                if self.take_profit and pnl_pct >= self.take_profit:
                    sell_tickers.append(ticker)
                    continue
                    
                # 檢查移動停損
                if self.trailing_stop:
                    drawdown = (current_price - high_price) / high_price
                    if drawdown <= self.trailing_stop:
                        sell_tickers.append(ticker)
                        continue
            
            if sell_tickers:
                # 賣出並進入等待買回狀態
                self.waiting_rebuy = True
                self.sold_date = dt
                
                result = {'_reason': '觸發停損停利，進入冷靜期'}
                for ticker in available_tickers:
                    if ticker in sell_tickers:
                        result[ticker] = 0  # 賣出
                return result
            
            return None  # 沒有需要調整的
        
        return None


class DCAStrategy(PortfolioStrategy):
    """
    定期定額策略
    
    每月固定日期買入，永不賣出（純累積模式）
    如果指定日期沒開盤，會在該月下一個交易日買入
    """
    def __init__(self, buy_day: int = 1, monthly_amount: float = None):
        super().__init__(name="DCA (定期定額)")
        self.buy_day = buy_day  # 每月第幾天買 (1-28 比較安全)
        self.monthly_amount = monthly_amount or 10000  # 每月投入金額
        self.last_buy_month = None
        self.is_first_buy = True  # 追蹤是否為第一次購買
        self.buy_only = True  # 標記這是純買入策略，不允許賣出
        
    def rebalance_signal(self, current_date, available_tickers: list, 
                         data_slice: dict, positions: dict = None) -> dict:
        """
        每月指定日期買入，永不賣出
        
        邏輯：
        1. 第一天有資料就買入（初始資金）
        2. 之後每個月，在 >= buy_day 的第一個交易日買入
           （如果 buy_day 是假日，會在下一個交易日買入）
        """
        if isinstance(current_date, str):
            try:
                dt = datetime.strptime(current_date, '%Y-%m-%d')
            except:
                dt = datetime.strptime(current_date[:10], '%Y-%m-%d')
        else:
            dt = current_date
            
        current_month = (dt.year, dt.month)
        
        # 這個月已經買過了，跳過
        if self.last_buy_month == current_month:
            return None
        
        # 第一次購買：立即執行（使用初始資金）
        if self.is_first_buy:
            pass  # 直接往下執行買入
        else:
            # 非第一次：檢查日期是否 >= buy_day
            if dt.day < self.buy_day:
                return None
        
        # 標記這個月已買入
        self.last_buy_month = current_month
        
        # DCA 使用特殊標記回傳，告訴引擎要用固定金額買入
        if not available_tickers:
            return {}
        
        # 回傳特殊結構，包含買入金額
        result = {
            '_dca_mode': True,  # 標記 DCA 模式
            '_monthly_amount': self.monthly_amount,
            '_is_first_buy': self.is_first_buy
        }
        
        # 等權重分配到各股票
        weight = 1.0 / len(available_tickers)
        for ticker in available_tickers:
            result[ticker] = weight
            
        self.is_first_buy = False
        return result


class StrategyDrivenPortfolio(PortfolioStrategy):
    """
    策略驅動型投資組合
    
    每檔股票用指定的技術策略來決定進出場
    """
    def __init__(self, stock_strategies: Dict[str, str] = None,
                 default_strategy: str = "MA5x20"):
        super().__init__(name="Strategy Driven")
        self.stock_strategies = stock_strategies or {}
        self.default_strategy = default_strategy
        self.positions = {}  # {ticker: True/False} 是否持有
        
    def _check_signal(self, ticker: str, row: dict, strategy: str) -> int:
        """
        檢查策略訊號
        
        Returns:
            1 = 買入, -1 = 賣出, 0 = 觀望
        """
        # MA 策略
        if strategy == "MA5x20":
            ma5 = row.get('ma5') or row.get('MA5')
            ma20 = row.get('ma20') or row.get('MA20')
            if ma5 and ma20:
                if ma5 > ma20:
                    return 1 if not self.positions.get(ticker) else 0
                elif ma5 < ma20:
                    return -1 if self.positions.get(ticker) else 0
                    
        elif strategy == "MA5x60":
            ma5 = row.get('ma5') or row.get('MA5')
            ma60 = row.get('ma60') or row.get('MA60')
            if ma5 and ma60:
                if ma5 > ma60:
                    return 1 if not self.positions.get(ticker) else 0
                elif ma5 < ma60:
                    return -1 if self.positions.get(ticker) else 0
                    
        elif strategy == "MACD":
            macd = row.get('macd') or row.get('MACD')
            signal = row.get('macd_signal') or row.get('signal')
            if macd is not None and signal is not None:
                if macd > signal:
                    return 1 if not self.positions.get(ticker) else 0
                elif macd < signal:
                    return -1 if self.positions.get(ticker) else 0
                    
        elif strategy == "RSI":
            rsi = row.get('rsi') or row.get('RSI')
            if rsi is not None:
                if rsi < 30:
                    return 1 if not self.positions.get(ticker) else 0
                elif rsi > 70:
                    return -1 if self.positions.get(ticker) else 0
                    
        elif strategy == "外資連買":
            foreign_5d = row.get('foreign_5d', 0)
            foreign = row.get('foreign', 0)
            if foreign_5d > 0:  # 5日外資累計為正
                return 1 if not self.positions.get(ticker) else 0
            elif foreign < 0:
                return -1 if self.positions.get(ticker) else 0
                
        elif strategy == "投信連買":
            trust_5d = row.get('trust_5d', 0)
            trust = row.get('trust', 0)
            if trust_5d > 0:
                return 1 if not self.positions.get(ticker) else 0
            elif trust < 0:
                return -1 if self.positions.get(ticker) else 0
        
        elif strategy == "買入持有":
            # 第一次遇到就買入
            if not self.positions.get(ticker):
                return 1
            return 0
        
        return 0
        
    def rebalance_signal(self, current_date, available_tickers: list, 
                         data_slice: dict, positions: dict = None) -> dict:
        """根據每檔股票的策略產生訊號"""
        result = {}
        active_count = 0
        
        for ticker in available_tickers:
            if ticker not in data_slice:
                continue
                
            row = data_slice[ticker]
            strategy = self.stock_strategies.get(ticker, self.default_strategy)
            signal = self._check_signal(ticker, row, strategy)
            
            if signal == 1:  # 買入
                self.positions[ticker] = True
                result[ticker] = 1.0 / len(available_tickers)  # 臨時權重
                active_count += 1
            elif signal == -1:  # 賣出
                self.positions[ticker] = False
                result[ticker] = 0
                
        # 重新計算權重（只分配給要持有的股票）
        if active_count > 0:
            weight = 1.0 / active_count
            for ticker in result:
                if result[ticker] > 0:
                    result[ticker] = weight
        
        return result if result else None
