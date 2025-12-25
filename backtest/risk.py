# -*- coding: utf-8 -*-
"""
風險管理模組
"""
import numpy as np

class PositionSizer:
    """
    部位規模計算器
    """
    def __init__(self, method='percent', value=0.1):
        """
        初始化
        
        Args:
            method: 計算方式
                - 'percent': 佔總資金百分比 (0.1 = 10%)
                - 'fixed_amount': 固定金額 (100000 = 10萬)
                - 'kelly': 凱利公式 (需提供勝率與賠率)
            value: 對應數值
        """
        self.method = method
        self.value = value

    def get_shares(self, capital: float, price: float, win_rate: float = 0, profit_factor: float = 0) -> int:
        """
        計算應買入股數
        
        Args:
            capital: 目前可用資金
            price: 股票價格
            win_rate: 策略預期勝率 (凱利公式用)
            profit_factor: 策略預期賠率 (凱利公式用)
            
        Returns:
            int: 應買入股數
        """
        if price <= 0:
            return 0
            
        target_amount = 0
        
        if self.method == 'percent':
            target_amount = capital * self.value
            
        elif self.method == 'fixed_amount':
            target_amount = min(capital, self.value)
            
        elif self.method == 'kelly':
            if win_rate <= 0 or profit_factor <= 0:
                print("⚠️ 警告: 凱利公式需要勝率與賠率 > 0，退回預設 10%")
                target_amount = capital * 0.1
            else:
                # Kelly = W - (1-W)/R
                kelly_fraction = win_rate - (1 - win_rate) / profit_factor
                # 通常建議用 Half Kelly 避免風險過大
                kelly_fraction = max(0, kelly_fraction * 0.5)
                # 上限 50%
                kelly_fraction = min(0.5, kelly_fraction)
                target_amount = capital * kelly_fraction
        
        # 計算股數 (無條件捨去)
        shares = int(target_amount // price)
        return shares


class RiskManager:
    """
    風險管理器 (停損/停利)
    """
    def __init__(self, stop_loss_pct=None, take_profit_pct=None, trailing_stop_pct=None):
        """
        初始化
        
        Args:
            stop_loss_pct: 固定停損 % (e.g. 0.1 = -10%)
            take_profit_pct: 固定停利 %
            trailing_stop_pct: 移動停損 %
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        
        # 狀態追蹤
        self.highest_price = 0
        
    def reset(self):
        """重置狀態 (新交易開始)"""
        self.highest_price = 0
        
    def check_exit(self, entry_price: float, current_price: float, current_date=None, verbose=False) -> str:
        """
        檢查是否觸發出場條件
        
        Returns:
            str: 出場原因 ('STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP', None)
        """
        if entry_price <= 0:
            return None
            
        # 更新最高價 (移動停損用)
        if current_price > self.highest_price:
            self.highest_price = current_price
            
        # 1. 固定停損
        if self.stop_loss_pct:
            loss_threshold = entry_price * (1 - self.stop_loss_pct)
            if current_price <= loss_threshold:
                if verbose:
                    print(f"🛑 觸發停損 ({current_date}): 現價 {current_price} <= 閾值 {loss_threshold:.2f}")
                return 'STOP_LOSS'
                
        # 2. 固定停利
        if self.take_profit_pct:
            profit_threshold = entry_price * (1 + self.take_profit_pct)
            if current_price >= profit_threshold:
                if verbose:
                    print(f"💰 觸發停利 ({current_date}): 現價 {current_price} >= 閾值 {profit_threshold:.2f}")
                return 'TAKE_PROFIT'
                
        # 3. 移動停損
        if self.trailing_stop_pct and self.highest_price > 0:
            # 從最高點回落超過 N%
            trailing_threshold = self.highest_price * (1 - self.trailing_stop_pct)
            if current_price <= trailing_threshold:
                if verbose:
                    print(f"📉 觸發移動停損 ({current_date}): 最高 {self.highest_price} -> 現價 {current_price} (跌破 {trailing_threshold:.2f})")
                return 'TRAILING_STOP'
                
        return None
