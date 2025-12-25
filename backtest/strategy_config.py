# -*- coding: utf-8 -*-
"""
策略配置模組
定義策略的進出場條件和風險管理參數
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum


class StrategyType(Enum):
    """策略類型"""
    MA5x20 = "MA5x20"
    MA5x60 = "MA5x60"
    MACD = "MACD"
    RSI = "RSI"
    MOMENTUM = "動量突破"
    TURTLE = "海龜策略"
    FOREIGN = "外資連買"
    TRUST = "投信連買"
    BUY_HOLD = "買入持有"
    DCA = "定期定額"  # Dollar Cost Averaging


class ExitType(Enum):
    """出場類型"""
    STRATEGY = "strategy"      # 策略訊號出場
    STOP_LOSS = "stop_loss"    # 停損出場
    TAKE_PROFIT = "take_profit"  # 停利出場
    TRAILING_STOP = "trailing_stop"  # 移動停損出場
    HOLD_FOREVER = "hold_forever"  # 持有到回測結束


@dataclass
class StrategyConfig:
    """
    策略配置
    
    定義一檔股票的進出場策略和風險管理參數
    """
    # 進場策略
    entry_strategy: str = "MA5x20"
    
    # 出場條件（可多選）
    exit_by_strategy: bool = True  # 策略訊號出場
    stop_loss: Optional[float] = None  # 停損比例 (如 -0.10 = -10%)
    take_profit: Optional[float] = None  # 停利比例 (如 0.30 = +30%)
    trailing_stop: Optional[float] = None  # 移動停損比例 (如 -0.10 = 從高點回落10%)
    
    # 資金配置
    weight: float = 1.0  # 權重（等權重時會自動計算）
    
    def __post_init__(self):
        """驗證參數"""
        if self.stop_loss is not None and self.stop_loss >= 0:
            self.stop_loss = -abs(self.stop_loss)  # 確保是負數
        if self.take_profit is not None and self.take_profit <= 0:
            self.take_profit = abs(self.take_profit)  # 確保是正數
        if self.trailing_stop is not None and self.trailing_stop >= 0:
            self.trailing_stop = -abs(self.trailing_stop)  # 確保是負數


@dataclass
class PortfolioConfig:
    """
    投資組合配置
    
    定義整個投資組合的策略設定
    """
    # 股票策略配置 {ticker: StrategyConfig}
    stock_configs: Dict[str, StrategyConfig] = field(default_factory=dict)
    
    # 再平衡設定
    rebalance_mode: str = "strategy"  # strategy, monthly, never
    
    # 預設出場條件（當個股沒有設定時使用）
    default_stop_loss: Optional[float] = None
    default_take_profit: Optional[float] = None
    default_trailing_stop: Optional[float] = None
    
    def add_stock(self, ticker: str, 
                  strategy: str = "MA5x20",
                  stop_loss: float = None,
                  take_profit: float = None,
                  trailing_stop: float = None) -> None:
        """新增股票配置"""
        # 使用預設值
        sl = stop_loss if stop_loss is not None else self.default_stop_loss
        tp = take_profit if take_profit is not None else self.default_take_profit
        ts = trailing_stop if trailing_stop is not None else self.default_trailing_stop
        
        self.stock_configs[ticker] = StrategyConfig(
            entry_strategy=strategy,
            stop_loss=sl,
            take_profit=tp,
            trailing_stop=ts
        )
    
    def get_tickers(self) -> List[str]:
        """取得所有股票代碼"""
        return list(self.stock_configs.keys())


# ========== 策略說明 ==========

STRATEGY_DESCRIPTIONS = {
    "MA5x20": {
        "name": "MA5x20 均線交叉",
        "entry": "5日均線向上穿越20日均線（黃金交叉）",
        "exit": "5日均線向下穿越20日均線（死亡交叉）",
        "type": "趨勢追蹤",
        "risk": "中"
    },
    "MA5x60": {
        "name": "MA5x60 中期均線",
        "entry": "5日均線向上穿越60日均線",
        "exit": "5日均線向下穿越60日均線",
        "type": "中期趨勢",
        "risk": "中低"
    },
    "MACD": {
        "name": "MACD 動能策略",
        "entry": "MACD線向上穿越信號線（快線上穿慢線）",
        "exit": "MACD線向下穿越信號線",
        "type": "動能策略",
        "risk": "中"
    },
    "RSI": {
        "name": "RSI 超買超賣",
        "entry": "RSI < 30（超賣區）",
        "exit": "RSI > 70（超買區）",
        "type": "反轉策略",
        "risk": "中高"
    },
    "動量突破": {
        "name": "動量突破策略",
        "entry": "股價突破20日最高價",
        "exit": "股價跌破20日最低價",
        "type": "突破策略",
        "risk": "高"
    },
    "布林通道": {
        "name": "布林通道策略",
        "entry": "股價由下碰觸布林下軌並反彈",
        "exit": "股價由上碰觸布林上軌並回落",
        "type": "波動策略",
        "risk": "中"
    },
    "量價突破": {
        "name": "量價突破策略",
        "entry": "成交量暴增2倍 + 股價漲2%",
        "exit": "成交量萎縮 + 股價跌2%",
        "type": "突破策略",
        "risk": "高"
    },
    "海龜策略": {
        "name": "海龜交易策略",
        "entry": "股價突破55日最高價",
        "exit": "股價跌破20日最低價",
        "type": "長期趨勢",
        "risk": "中"
    },
    "外資連買": {
        "name": "外資連續買超",
        "entry": "外資連續買超5天",
        "exit": "外資連續賣超5天",
        "type": "籌碼策略",
        "risk": "中"
    },
    "外資連買3天": {
        "name": "外資連買3天",
        "entry": "外資連續買超3天",
        "exit": "外資連續賣超3天",
        "type": "籌碼策略",
        "risk": "中高"
    },
    "外資連買5天": {
        "name": "外資連買5天",
        "entry": "外資連續買超5天",
        "exit": "外資連續賣超5天",
        "type": "籌碼策略",
        "risk": "中"
    },
    "投信連買": {
        "name": "投信連續買超",
        "entry": "投信連續買超5天",
        "exit": "投信連續賣超5天",
        "type": "籌碼策略",
        "risk": "中"
    },
    "投信連買3天": {
        "name": "投信連買3天",
        "entry": "投信連續買超3天",
        "exit": "投信連續賣超3天",
        "type": "籌碼策略",
        "risk": "中高"
    },
    "投信連買5天": {
        "name": "投信連買5天",
        "entry": "投信連續買超5天",
        "exit": "投信連續賣超5天",
        "type": "籌碼策略",
        "risk": "中"
    },
    "買入持有": {
        "name": "買入持有策略",
        "entry": "回測開始時買入",
        "exit": "根據設定的停損/停利/回測結束",
        "type": "長期投資",
        "risk": "視標的而定"
    },
    "定期定額": {
        "name": "定期定額策略",
        "entry": "每月1號買入固定金額",
        "exit": "回測結束時結算",
        "type": "長期投資",
        "risk": "低"
    }
}


def get_strategy_description(strategy_name: str) -> dict:
    """取得策略說明"""
    return STRATEGY_DESCRIPTIONS.get(strategy_name, {
        "name": strategy_name,
        "entry": "未定義",
        "exit": "未定義",
        "type": "未知",
        "risk": "未知"
    })
