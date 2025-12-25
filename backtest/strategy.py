# -*- coding: utf-8 -*-
"""
策略基類與內建策略
"""
import pandas as pd
import numpy as np


class Strategy:
    """
    策略基類
    
    用戶需繼承此類並實作 generate_signals() 方法
    """
    
    def __init__(self, name="MyStrategy"):
        self.name = name
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        產生交易訊號
        
        Args:
            df: 包含 OHLCV 和技術指標的 DataFrame
        
        Returns:
            pd.Series: 訊號序列
                1 = 買入
                0 = 持有/觀望
               -1 = 賣出
        """
        raise NotImplementedError("請實作 generate_signals 方法")
    
    def __repr__(self):
        return f"<Strategy: {self.name}>"


class MACrossStrategy(Strategy):
    """
    MA 均線交叉策略
    
    - 短均線上穿長均線 → 買入
    - 短均線下穿長均線 → 賣出
    """
    
    def __init__(self, short_period=5, long_period=20):
        super().__init__(name=f"MA{short_period}x{long_period}")
        self.short_period = short_period
        self.long_period = long_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        short_ma = f'ma{self.short_period}'
        long_ma = f'ma{self.long_period}'
        
        # 確保欄位存在
        if short_ma not in df.columns or long_ma not in df.columns:
            raise ValueError(f"DataFrame 需要包含 {short_ma} 和 {long_ma} 欄位")
        
        # 黃金交叉（短上穿長）→ 買入
        golden_cross = (df[short_ma] > df[long_ma]) & (df[short_ma].shift(1) <= df[long_ma].shift(1))
        signals[golden_cross] = 1
        
        # 死亡交叉（短下穿長）→ 賣出
        death_cross = (df[short_ma] < df[long_ma]) & (df[short_ma].shift(1) >= df[long_ma].shift(1))
        signals[death_cross] = -1
        
        return signals


class RSIStrategy(Strategy):
    """
    RSI 超買超賣策略
    
    - RSI < oversold (30) 時買入
    - RSI > overbought (70) 時賣出
    """
    
    def __init__(self, oversold=30, overbought=70):
        super().__init__(name=f"RSI({oversold},{overbought})")
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        if 'rsi' not in df.columns:
            raise ValueError("DataFrame 需要包含 rsi 欄位")
        
        # RSI 由下往上突破超賣區 → 買入
        buy_signal = (df['rsi'] > self.oversold) & (df['rsi'].shift(1) <= self.oversold)
        signals[buy_signal] = 1
        
        # RSI 由上往下跌破超買區 → 賣出
        sell_signal = (df['rsi'] < self.overbought) & (df['rsi'].shift(1) >= self.overbought)
        signals[sell_signal] = -1
        
        return signals


class KDStrategy(Strategy):
    """
    KD 隨機指標策略
    
    - K 線上穿 D 線（黃金交叉）→ 買入
    - K 線下穿 D 線（死亡交叉）→ 賣出
    """
    
    def __init__(self, oversold=20, overbought=80):
        super().__init__(name=f"KD({oversold},{overbought})")
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        if 'k' not in df.columns or 'd' not in df.columns:
            raise ValueError("DataFrame 需要包含 k 和 d 欄位")
        
        # K 上穿 D 且在超賣區 → 買入
        buy_signal = ((df['k'] > df['d']) & 
                      (df['k'].shift(1) <= df['d'].shift(1)) & 
                      (df['k'] < self.overbought))
        signals[buy_signal] = 1
        
        # K 下穿 D 且在超買區 → 賣出
        sell_signal = ((df['k'] < df['d']) & 
                       (df['k'].shift(1) >= df['d'].shift(1)) & 
                       (df['k'] > self.oversold))
        signals[sell_signal] = -1
        
        return signals


class MACDStrategy(Strategy):
    """
    MACD 策略
    
    - MACD 上穿信號線 → 買入
    - MACD 下穿信號線 → 賣出
    """
    
    def __init__(self):
        super().__init__(name="MACD")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            raise ValueError("DataFrame 需要包含 macd 和 macd_signal 欄位")
        
        # MACD 上穿信號線 → 買入
        buy_signal = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        signals[buy_signal] = 1
        
        # MACD 下穿信號線 → 賣出
        sell_signal = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        signals[sell_signal] = -1
        
        return signals


# ========== 法人跟單策略 ==========

class InstitutionalFollowStrategy(Strategy):
    """
    法人跟單策略
    
    跟著外資/投信/自營商買賣
    需要使用 data_loader.load_stock_with_institutional() 載入資料
    """
    
    def __init__(self, inst_type='foreign', consecutive_days=3, threshold=0):
        """
        Args:
            inst_type: 法人類型 ('foreign'=外資, 'trust'=投信, 'dealer'=自營商, 'inst_total'=三大法人)
            consecutive_days: 連續買超天數
            threshold: 買超門檻（張）
        """
        super().__init__(name=f"跟隨{inst_type}({consecutive_days}天)")
        self.inst_type = inst_type
        self.consecutive_days = consecutive_days
        self.threshold = threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        if self.inst_type not in df.columns:
            raise ValueError(f"DataFrame 需要包含 {self.inst_type} 欄位，請使用 load_stock_with_institutional() 載入資料")
        
        inst = df[self.inst_type]
        
        # 計算連續買超天數
        is_buying = inst > self.threshold
        consecutive_buys = is_buying.rolling(self.consecutive_days).sum()
        
        # 連續買超 N 天 → 買入
        buy_signal = (consecutive_buys == self.consecutive_days) & (consecutive_buys.shift(1) < self.consecutive_days)
        signals[buy_signal] = 1
        
        # 連續賣超 N 天 → 賣出
        is_selling = inst < -self.threshold
        consecutive_sells = is_selling.rolling(self.consecutive_days).sum()
        sell_signal = (consecutive_sells == self.consecutive_days) & (consecutive_sells.shift(1) < self.consecutive_days)
        signals[sell_signal] = -1
        
        return signals


class ChipTechStrategy(Strategy):
    """
    籌碼+技術綜合策略
    
    買入條件：法人買超 + RSI 超賣 + 技術指標確認
    賣出條件：法人賣超 + RSI 超買
    """
    
    def __init__(self, inst_type='foreign', rsi_oversold=30, rsi_overbought=70):
        super().__init__(name=f"籌碼技術({inst_type})")
        self.inst_type = inst_type
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        required = [self.inst_type, 'rsi', 'macd', 'macd_signal']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame 缺少欄位: {missing}")
        
        inst = df[self.inst_type]
        
        # 買入：法人買超 + RSI < 40 + MACD 金叉
        buy_condition = (
            (inst > 0) &
            (df['rsi'] < 40) &
            (df['macd'] > df['macd_signal']) &
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        signals[buy_condition] = 1
        
        # 賣出：法人賣超 + RSI > 60
        sell_condition = (
            (inst < 0) &
            (df['rsi'] > 60)
        )
        signals[sell_condition] = -1
        
        return signals


class MultiConditionStrategy(Strategy):
    """
    多條件策略（可自訂條件）
    
    使用範例:
        strategy = MultiConditionStrategy(
            buy_conditions=[
                ('rsi', '<', 30),
                ('macd', '>', 'macd_signal'),
                ('close', '>', 'ma20')
            ],
            sell_conditions=[
                ('rsi', '>', 70)
            ]
        )
    """
    
    def __init__(self, buy_conditions: list = None, sell_conditions: list = None, name="MultiCond"):
        super().__init__(name=name)
        self.buy_conditions = buy_conditions or []
        self.sell_conditions = sell_conditions or []
    
    def _evaluate_condition(self, df: pd.DataFrame, cond: tuple) -> pd.Series:
        """評估單一條件"""
        col, op, val = cond
        
        if col not in df.columns:
            raise ValueError(f"找不到欄位: {col}")
        
        # 如果 val 是欄位名稱，取該欄位
        if isinstance(val, str) and val in df.columns:
            compare_val = df[val]
        else:
            compare_val = val
        
        if op == '>':
            return df[col] > compare_val
        elif op == '<':
            return df[col] < compare_val
        elif op == '>=':
            return df[col] >= compare_val
        elif op == '<=':
            return df[col] <= compare_val
        elif op == '==':
            return df[col] == compare_val
        else:
            raise ValueError(f"不支援的運算子: {op}")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        # 評估買入條件（所有條件都要滿足）
        if self.buy_conditions:
            buy_mask = pd.Series(True, index=df.index)
            for cond in self.buy_conditions:
                buy_mask = buy_mask & self._evaluate_condition(df, cond)
            signals[buy_mask] = 1
        
        # 評估賣出條件
        if self.sell_conditions:
            sell_mask = pd.Series(True, index=df.index)
            for cond in self.sell_conditions:
                sell_mask = sell_mask & self._evaluate_condition(df, cond)
            signals[sell_mask] = -1
        
        return signals


class BollingerStrategy(Strategy):
    """
    布林通道策略
    
    - 股價跌破下軌 → 買入
    - 股價突破上軌 → 賣出
    """
    
    def __init__(self):
        super().__init__(name="Bollinger")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        required = ['close', 'bb_lower', 'bb_upper']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame 缺少欄位: {missing}")
        
        # 股價從下方穿越下軌 → 買入
        buy_signal = (df['close'] > df['bb_lower']) & (df['close'].shift(1) <= df['bb_lower'].shift(1))
        signals[buy_signal] = 1
        
        # 股價從上方穿越上軌 → 賣出
        sell_signal = (df['close'] < df['bb_upper']) & (df['close'].shift(1) >= df['bb_upper'].shift(1))
        signals[sell_signal] = -1
        
        return signals


class MomentumBreakoutStrategy(Strategy):
    """
    動量突破策略
    
    - 股價突破 N 日高點 + 成交量放大 → 買入
    - 股價跌破 N 日低點 → 賣出
    """
    
    def __init__(self, period=20, volume_mult=1.5):
        """
        Args:
            period: 突破週期（預設 20 日）
            volume_mult: 成交量倍數門檻（預設 1.5 倍均量）
        """
        super().__init__(name=f"動量突破({period}日)")
        self.period = period
        self.volume_mult = volume_mult
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        # 計算 N 日高點、低點
        high_n = df['high'].rolling(self.period).max()
        low_n = df['low'].rolling(self.period).min()
        
        # 成交量均線
        vol_ma = df['volume'].rolling(self.period).mean()
        
        # 突破高點 + 量增 → 買入
        buy_signal = (
            (df['close'] > high_n.shift(1)) &
            (df['volume'] > vol_ma * self.volume_mult)
        )
        signals[buy_signal] = 1
        
        # 跌破低點 → 賣出
        sell_signal = df['close'] < low_n.shift(1)
        signals[sell_signal] = -1
        
        return signals


class MeanReversionStrategy(Strategy):
    """
    均值回歸策略
    
    假設價格會回歸均值
    - 價格低於 MA 很多 + RSI 超賣 → 買入
    - 價格高於 MA 很多 + RSI 超買 → 賣出
    """
    
    def __init__(self, ma_period=20, deviation=0.05, rsi_low=30, rsi_high=70):
        """
        Args:
            ma_period: 均線週期
            deviation: 偏離度門檻（5%）
            rsi_low: RSI 超賣門檻
            rsi_high: RSI 超買門檻
        """
        super().__init__(name=f"均值回歸(MA{ma_period})")
        self.ma_period = ma_period
        self.deviation = deviation
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        ma_col = f'ma{self.ma_period}'
        if ma_col not in df.columns:
            raise ValueError(f"需要 {ma_col} 欄位")
        if 'rsi' not in df.columns:
            raise ValueError("需要 rsi 欄位")
        
        ma = df[ma_col]
        
        # 計算偏離度
        deviation_pct = (df['close'] - ma) / ma
        
        # 低於均線太多 + RSI 超賣 → 買入
        buy_signal = (deviation_pct < -self.deviation) & (df['rsi'] < self.rsi_low)
        signals[buy_signal] = 1
        
        # 高於均線太多 + RSI 超買 → 賣出
        sell_signal = (deviation_pct > self.deviation) & (df['rsi'] > self.rsi_high)
        signals[sell_signal] = -1
        
        return signals


class VolumeBreakoutStrategy(Strategy):
    """
    成交量突破策略
    
    - 成交量暴增 + 股價上漲 → 買入
    - 成交量萎縮 + 股價下跌 → 賣出
    """
    
    def __init__(self, volume_mult=2.0, price_change=0.02):
        """
        Args:
            volume_mult: 成交量倍數（預設 2 倍）
            price_change: 價格變動門檻（預設 2%）
        """
        super().__init__(name=f"量價突破({volume_mult}x)")
        self.volume_mult = volume_mult
        self.price_change = price_change
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        # 成交量均線
        vol_ma = df['volume'].rolling(20).mean()
        
        # 價格變動
        price_pct = df['close'].pct_change()
        
        # 量增價漲 → 買入
        buy_signal = (
            (df['volume'] > vol_ma * self.volume_mult) &
            (price_pct > self.price_change)
        )
        signals[buy_signal] = 1
        
        # 量縮價跌 → 賣出
        sell_signal = (
            (df['volume'] < vol_ma * 0.5) &
            (price_pct < -self.price_change)
        )
        signals[sell_signal] = -1
        
        return signals


class TurtleStrategy(Strategy):
    """
    海龜交易策略
    
    經典趨勢追蹤策略
    - 突破 20 日高點 → 買入
    - 跌破 10 日低點 → 賣出
    """
    
    def __init__(self, entry_period=20, exit_period=10):
        super().__init__(name=f"海龜({entry_period}/{exit_period})")
        self.entry_period = entry_period
        self.exit_period = exit_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        entry_high = df['high'].rolling(self.entry_period).max()
        exit_low = df['low'].rolling(self.exit_period).min()
        
        # 突破 N 日高點 → 買入
        buy_signal = df['close'] > entry_high.shift(1)
        signals[buy_signal] = 1
        
        # 跌破 M 日低點 → 賣出
        sell_signal = df['close'] < exit_low.shift(1)
        signals[sell_signal] = -1
        
        return signals
