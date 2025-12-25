# -*- coding: utf-8 -*-
"""
回測引擎模組
"""

from .strategy import (
    Strategy, 
    MACrossStrategy, 
    RSIStrategy, 
    KDStrategy, 
    MACDStrategy,
    BollingerStrategy,
    InstitutionalFollowStrategy,
    ChipTechStrategy,
    MultiConditionStrategy,
    MomentumBreakoutStrategy,
    MeanReversionStrategy,
    VolumeBreakoutStrategy,
    TurtleStrategy
)
from .engine import BacktestEngine, quick_backtest
from .metrics import calculate_metrics, print_metrics
from .batch import batch_backtest, market_scan, compare_strategies
from .report import generate_html_report, print_summary

__all__ = [
    # 基礎策略
    'Strategy',
    'MACrossStrategy', 
    'RSIStrategy',
    'KDStrategy',
    'MACDStrategy',
    'BollingerStrategy',
    # 籌碼策略
    'InstitutionalFollowStrategy',
    'ChipTechStrategy',
    'MultiConditionStrategy',
    # 進階策略
    'MomentumBreakoutStrategy',
    'MeanReversionStrategy',
    'VolumeBreakoutStrategy',
    'TurtleStrategy',
    # 引擎
    'BacktestEngine',
    'quick_backtest',
    # 批次回測
    'batch_backtest',
    'market_scan',
    'compare_strategies',
    # 報表
    'generate_html_report',
    'print_summary',
    # 指標
    'calculate_metrics',
    'print_metrics'
]
