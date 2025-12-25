#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================
📊 台股量化回測工具 - 簡易操作介面
========================================

這是一個簡單的命令列介面，讓你快速了解和使用回測功能。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from glob import glob

# 確保目錄存在
os.makedirs('reports', exist_ok=True)


def print_header():
    print("\n" + "=" * 60)
    print("📊 台股量化回測工具")
    print("=" * 60)


def print_menu():
    print("""
可用功能：
    1. 查看可用策略
    2. 單股票回測
    3. 多股票比較
    4. 全市場掃描（需時較久）
    5. 參數優化
    6. 查看報告目錄
    7. 什麼是夏普指數？
    8. 這個工具能幫我什麼？
    0. 離開
""")


def show_strategies():
    """顯示所有可用策略"""
    print("""
📋 可用策略列表：

【基礎策略】
    MACrossStrategy(5, 20)       - MA 均線交叉（短均線 5 日，長均線 20 日）
    RSIStrategy(30, 70)          - RSI 超買超賣（30 以下買，70 以上賣）
    MACDStrategy()               - MACD 金叉死叉
    BollingerStrategy()          - 布林通道突破

【籌碼策略】（需用 load_stock_with_institutional 載入資料）
    InstitutionalFollowStrategy('foreign', 3)  - 外資連續 3 天買超跟單
    ChipTechStrategy('foreign')                - 外資買超 + RSI + MACD

【進階策略】
    MomentumBreakoutStrategy(20) - 動量突破：突破 20 日高點買入
    VolumeBreakoutStrategy(2.0)  - 量價突破：成交量暴增 2 倍買入
    TurtleStrategy(20, 10)       - 海龜策略：經典趨勢追蹤
    MeanReversionStrategy(20)    - 均值回歸：偏離均線太多時反向操作

【自訂策略】
    MultiConditionStrategy(      - 多條件組合
        buy_conditions=[('rsi', '<', 30), ('close', '>', 'ma20')],
        sell_conditions=[('rsi', '>', 70)]
    )
""")


def single_stock_backtest():
    """單股票回測"""
    from backtest import BacktestEngine, MACrossStrategy
    from backtest.report import print_summary, generate_html_report
    
    ticker = input("\n請輸入股票代碼（如 2330.TW）: ").strip()
    
    # 找檔案
    files = glob(f'data/tw-share/dayK/{ticker}*.csv')
    if not files:
        print(f"❌ 找不到 {ticker} 的資料")
        return
    
    df = pd.read_csv(files[0])
    name = os.path.basename(files[0]).replace('.csv', '').split('_', 1)[-1]
    
    print(f"\n✅ 找到: {ticker} - {name}")
    print(f"   資料天數: {len(df)} 天")
    
    # 選擇策略
    print("\n選擇策略：")
    print("  1. MA5x20  2. MA5x60  3. RSI  4. MACD  5. 海龜策略")
    choice = input("請選擇 (1-5): ").strip()
    
    strategies = {
        '1': MACrossStrategy(5, 20),
        '2': MACrossStrategy(5, 60),
        '3': __import__('backtest').RSIStrategy(30, 70),
        '4': __import__('backtest').MACDStrategy(),
        '5': __import__('backtest').TurtleStrategy(20, 10),
    }
    
    strategy = strategies.get(choice, MACrossStrategy(5, 20))
    
    engine = BacktestEngine()
    result = engine.run(df, strategy)
    
    # 顯示結果
    print_summary(result)
    
    # 儲存報告
    report_path = f'reports/{ticker}_report.html'
    generate_html_report(result, ticker=f"{ticker} {name}", save_path=report_path)


def multi_stock_compare():
    """多股票比較"""
    from backtest import batch_backtest, MACrossStrategy
    
    print("\n熱門股票代碼：")
    print("  2330.TW(台積電) 2317.TW(鴻海) 2454.TW(聯發科)")
    print("  2882.TW(國泰金) 0050.TW(台灣50) 0056.TW(高股息)")
    
    tickers_input = input("\n請輸入股票代碼（用空格分隔）: ").strip()
    tickers = tickers_input.split()
    
    if not tickers:
        tickers = ['2330.TW', '2317.TW', '0050.TW']
    
    print(f"\n測試 {len(tickers)} 支股票...")
    
    strategy = MACrossStrategy(5, 20)
    results = batch_backtest(tickers=tickers, strategy=strategy, top_n=50)
    
    print("\n📈 結果（按報酬率排序）:")
    cols = ['ticker', 'name', 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    print(results[cols].to_string())


def run_market_scan():
    """全市場掃描"""
    print("\n⚠️ 全市場掃描需要 2-3 分鐘...")
    confirm = input("確定執行？(y/n): ").strip().lower()
    
    if confirm == 'y':
        os.system(f'{sys.executable} scan_market.py')
        print("\n📄 報告位置: reports/market_scan_all_strategies.html")


def optimize_strategy():
    """參數優化"""
    from backtest import BacktestEngine, MACrossStrategy
    
    ticker = input("\n請輸入股票代碼（如 2330.TW）: ").strip()
    
    files = glob(f'data/tw-share/dayK/{ticker}*.csv')
    if not files:
        print(f"❌ 找不到 {ticker} 的資料")
        return
    
    df = pd.read_csv(files[0])
    engine = BacktestEngine()
    
    print("\n🔍 優化 MA 策略參數...")
    result = engine.optimize(
        df, 
        MACrossStrategy,
        param_grid={
            'short_period': [5, 10, 20],
            'long_period': [20, 60, 120]
        },
        metric='sharpe_ratio',
        verbose=True
    )
    
    print("\n📊 所有結果：")
    cols = ['short_period', 'long_period', 'total_return', 'sharpe_ratio']
    print(result['all_results'][cols].sort_values('sharpe_ratio', ascending=False).to_string())


def explain_sharpe():
    """解釋夏普指數"""
    print("""
📖 什麼是夏普指數（Sharpe Ratio）？

【簡單說】
夏普指數 = 報酬 ÷ 風險

它告訴你：「每承擔 1 單位的風險，能獲得多少報酬」

【判斷標準】
    < 0    ：虧損，這策略不行
    0 ~ 1  ：報酬低於風險，普通
    1 ~ 2  ：不錯，報酬大於風險
    2 ~ 3  ：很好，值得使用
    > 3    ：優秀，但要注意是否過度擬合

【舉例】
    策略 A：報酬 50%，波動 25% → 夏普 = 2.0（好）
    策略 B：報酬 50%，波動 50% → 夏普 = 1.0（普通）
    
    雖然報酬一樣，但 A 策略風險更低，所以更好！

【重點】
    夏普高 = 穩定獲利
    夏普低 = 大起大落
""")


def explain_tool():
    """解釋這個工具能幫什麼"""
    print("""
🤔 這個工具能幫我什麼？

【1. 驗證投資想法】
    你想到一個策略，例如「RSI 低於 30 買入」
    -> 用這個工具測試過去一年這樣做會賺還是賠

【2. 找到最佳策略參數】
    不確定 MA 要用 5 日還是 20 日？
    -> 參數優化功能會自動找出最好的組合

【3. 篩選適合的股票】
    不是所有策略都適合所有股票
    -> 全市場掃描告訴你哪些股票適合哪個策略

【4. 評估風險】
    策略可能賺錢，但風險太大你睡不著
    -> 看最大回撤，知道最壞情況會虧多少

【5. 跟著法人操作】
    想跟外資一起買？
    -> 法人跟單策略測試這樣做的效果

【實際應用建議】

✅ 簡單開始：
    用 MA5x20 策略掃描市場
    找出夏普 > 2 的股票
    這些就是適合短線操作的股票

✅ 進階應用：
    用外資跟單策略
    當外資連續 3 天買超，你也跟著買
    測試顯示台積電這樣做報酬 103%

✅ 謹慎使用：
    過去績效不保證未來
    一定要看「最大回撤」評估最壞情況
    夏普 > 3 的結果可能是過度擬合

【下一步建議】
    1. 先用「單股票回測」測試你常買的股票
    2. 找出適合它的策略
    3. 用「全市場掃描」找到更多類似的機會
""")


def show_reports():
    """顯示報告目錄"""
    reports = glob('reports/*.html')
    if reports:
        print("\n📁 已產生的報告：")
        for r in reports:
            print(f"   {r}")
    else:
        print("\n📁 還沒有報告，請先執行回測")


def main():
    print_header()
    
    while True:
        print_menu()
        choice = input("請選擇功能 (0-8): ").strip()
        
        if choice == '0':
            print("\n👋 再見！")
            break
        elif choice == '1':
            show_strategies()
        elif choice == '2':
            single_stock_backtest()
        elif choice == '3':
            multi_stock_compare()
        elif choice == '4':
            run_market_scan()
        elif choice == '5':
            optimize_strategy()
        elif choice == '6':
            show_reports()
        elif choice == '7':
            explain_sharpe()
        elif choice == '8':
            explain_tool()
        else:
            print("❌ 無效選項，請重新選擇")
        
        input("\n按 Enter 繼續...")


if __name__ == '__main__':
    main()
