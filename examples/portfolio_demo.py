
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# 添加專案根目錄到 sys.path，確保能匯入模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.portfolio import PortfolioEngine
from backtest.strategy_portfolio import EqualWeightStrategy
from backtest.metrics import print_metrics
from backtest.portfolio_report import generate_portfolio_html_report, export_trades_csv

def load_data(tickers, base_dir):
    data_map = {}
    for ticker in tickers:
        file_path = os.path.join(base_dir, f"{ticker}.csv")
        # 這裡需要模糊匹配，因為檔名有中文
        # 簡單起見，我們假設我知道確切檔名，或者用 glob 搜尋
        # 稍微改寫一下，用 glob 找
        from glob import glob
        search_pattern = os.path.join(base_dir, f"{ticker}*.csv")
        files = glob(search_pattern)
        
        if not files:
            print(f"⚠️ 找不到 {ticker} 的資料")
            continue
            
        # 取第一個符合的
        path = files[0]
        print(f"📄 載入 {os.path.basename(path)}...")
        df = pd.read_csv(path)
        
        # 確保日期欄位正確
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)
            
        # 必須排序
        df = df.sort_values('date').reset_index(drop=True)
        data_map[ticker] = df
        
    return data_map

def main():
    # 1. 設定測試股票
    target_tickers = ['2330.TW', '2317.TW', '2454.TW'] # 台積電, 鴻海, 聯發科
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'tw-share', 'dayK')
    
    # 2. 載入資料
    print("📥 載入股票資料中...")
    data_map = load_data(target_tickers, data_dir)
    
    if not data_map:
        print("❌ 無資料，結束測試")
        return

    # 3. 初始化引擎與策略
    engine = PortfolioEngine(initial_capital=1_000_000)
    strategy = EqualWeightStrategy() # 等權重
    
    # 4. 執行回測
    print("\n🚀 開始執行投資組合回測 (Equal Weight)...")
    result = engine.run(data_map, strategy)
    
    # 5. 顯示結果
    print("\n" + "="*50)
    print("📊 回測結果報告")
    print("="*50)
    
    metrics = result['metrics']
    print_metrics(metrics)
    
    # 顯示持倉變化 (前 5 天與後 5 天)
    positions = result['positions']
    print(f"\n最終持倉狀况: {positions}")
    
    # 6. 輸出報表
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
    
    # HTML 報表
    html_path = os.path.join(report_dir, 'portfolio_report.html')
    generate_portfolio_html_report(result, tickers=target_tickers, strategy_name='Equal Weight', save_path=html_path)
    
    # CSV 交易明細
    csv_path = os.path.join(report_dir, 'portfolio_trades.csv')
    export_trades_csv(result, csv_path)
    
    print("\n✅ 回測完成！")
    print(f"📄 報表路徑: {html_path}")

if __name__ == "__main__":
    main()
