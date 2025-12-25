# -*- coding: utf-8 -*-
"""
========================================
📊 全市場策略掃描工具 (夏普比率掃描)
========================================

功能：
1. 掃描全市場股票，用 10 種策略回測
2. 根據「夏普比率」排名，找出適合各策略的股票
3. 產生「跨策略總排名」綜合推薦股票

策略說明：
- 技術分析策略 (8 種): MA5x20, MA5x60, RSI, MACD, 布林通道, 動量突破, 量價突破, 海龜策略
- 法人籌碼策略 (2 種): 外資連買, 投信連買

用法：
    python scan_market.py

報告輸出：
    reports/market_scan_all_strategies.html
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from glob import glob
from datetime import datetime
from tqdm import tqdm

from backtest import (
    BacktestEngine,
    MACrossStrategy,
    RSIStrategy,
    MACDStrategy,
    BollingerStrategy,
    MomentumBreakoutStrategy,
    VolumeBreakoutStrategy,
    TurtleStrategy,
    InstitutionalFollowStrategy,
)
from data_loader import load_institutional_data, load_stock_with_institutional

# 載入法人資料
INSTITUTIONAL_DATA = None
try:
    INSTITUTIONAL_DATA = load_institutional_data()
    print(f"✅ 已載入法人資料: {len(INSTITUTIONAL_DATA)} 天")
except:
    print("⚠️ 無法載入法人資料，法人策略將跳過")

# 資料目錄
STOCK_DIR = os.path.join(os.path.dirname(__file__), "data", "tw-share", "dayK")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def get_all_strategies(include_institutional=True):
    """取得所有可用策略"""
    strategies = [
        ("MA5x20", MACrossStrategy(5, 20)),
        ("MA5x60", MACrossStrategy(5, 60)),
        ("RSI", RSIStrategy(30, 70)),
        ("MACD", MACDStrategy()),
        ("布林通道", BollingerStrategy()),
        ("動量突破", MomentumBreakoutStrategy(20)),
        ("量價突破", VolumeBreakoutStrategy(2.0)),
        ("海龜策略", TurtleStrategy(20, 10)),
    ]
    
    if include_institutional and INSTITUTIONAL_DATA is not None:
        strategies.extend([
            ("外資連買", InstitutionalFollowStrategy('foreign', 3, threshold=100)),
            ("投信連買", InstitutionalFollowStrategy('trust', 3, threshold=10)),  # 降低門檻
        ])
    
    return strategies


def compute_overall_ranking(results: dict, top_n=30):
    """
    計算跨策略總排名
    
    將各策略的結果彙整，計算每支股票的「綜合分數」：
    - 出現在越多策略中 → 分數越高
    - 平均夏普比率越高 → 分數越高
    
    公式: 綜合分數 = 出現策略數 × 平均夏普比率
    """
    stock_stats = {}  # {ticker: {name, strategies: [], sharpe_list: [], ...}}
    
    for strategy_name, df in results.items():
        if df.empty:
            continue
        
        for _, row in df.iterrows():
            ticker = row['ticker']
            if ticker not in stock_stats:
                stock_stats[ticker] = {
                    'name': row['name'],
                    'strategies': [],
                    'sharpe_list': [],
                    'return_list': [],
                    'best_sharpe': 0,
                    'best_strategy': '',
                }
            
            stock_stats[ticker]['strategies'].append(strategy_name)
            stock_stats[ticker]['sharpe_list'].append(row['sharpe_ratio'])
            stock_stats[ticker]['return_list'].append(row['total_return'])
            
            if row['sharpe_ratio'] > stock_stats[ticker]['best_sharpe']:
                stock_stats[ticker]['best_sharpe'] = row['sharpe_ratio']
                stock_stats[ticker]['best_strategy'] = strategy_name
    
    # 計算綜合分數
    ranking_data = []
    for ticker, stats in stock_stats.items():
        strategy_count = len(stats['strategies'])
        avg_sharpe = sum(stats['sharpe_list']) / strategy_count
        avg_return = sum(stats['return_list']) / strategy_count
        
        # 綜合分數 = 策略數 × 平均夏普比率
        score = strategy_count * avg_sharpe
        
        ranking_data.append({
            'ticker': ticker,
            'name': stats['name'],
            'score': score,
            'strategy_count': strategy_count,
            'avg_sharpe': avg_sharpe,
            'avg_return': avg_return,
            'best_strategy': stats['best_strategy'],
            'best_sharpe': stats['best_sharpe'],
            'strategies': ', '.join(stats['strategies'][:3]) + ('...' if strategy_count > 3 else ''),
        })
    
    # 排序並取前 N
    ranking_df = pd.DataFrame(ranking_data)
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values('score', ascending=False).head(top_n)
    
    return ranking_df


def market_scan_all_strategies(top_n=30, min_volume=500):
    """
    全市場掃描所有策略
    
    Returns:
        dict: {strategy_name: DataFrame}
    """
    files = glob(os.path.join(STOCK_DIR, "*.csv"))
    strategies = get_all_strategies()
    engine = BacktestEngine()
    
    results = {name: [] for name, _ in strategies}
    
    print(f"🔍 全市場掃描")
    print(f"   股票數: {len(files)} 檔")
    print(f"   策略數: {len(strategies)} 種")
    print()
    
    for csv_path in tqdm(files, desc="掃描中"):
        try:
            df = pd.read_csv(csv_path)
            
            # 過濾成交量太低的
            if df['volume'].mean() < min_volume:
                continue
            
            ticker = os.path.basename(csv_path).split('_')[0]
            name = os.path.basename(csv_path).replace('.csv', '').split('_', 1)[-1]
            
            # 合併法人資料（如果有）
            df_with_inst = None
            if INSTITUTIONAL_DATA is not None:
                try:
                    df_with_inst = load_stock_with_institutional(ticker)
                except:
                    pass
            
            for strategy_name, strategy in strategies:
                try:
                    # 法人策略使用合併後的資料
                    if '連買' in strategy_name or '連賣' in strategy_name:
                        if df_with_inst is None or df_with_inst.empty:
                            continue
                        run_df = df_with_inst
                    else:
                        run_df = df
                    
                    result = engine.run(run_df, strategy, verbose=False)
                    m = result['metrics']
                    
                    if m['trade_count'] >= 3:  # 至少 3 筆交易
                        results[strategy_name].append({
                            'ticker': ticker,
                            'name': name,
                            'total_return': m['total_return'],
                            'sharpe_ratio': m['sharpe_ratio'],
                            'max_drawdown': m['max_drawdown'],
                            'win_rate': m['win_rate'],
                            'trade_count': m['trade_count']
                        })
                except:
                    continue
                    
        except:
            continue
    
    # 轉換為 DataFrame 並排序
    for name in results:
        if results[name]:
            df = pd.DataFrame(results[name])
            results[name] = df.sort_values('sharpe_ratio', ascending=False).head(top_n)
        else:
            results[name] = pd.DataFrame()
    
    # 計算跨策略總排名
    overall_ranking = compute_overall_ranking(results)
    
    return results, overall_ranking


def generate_scan_report(results: dict, overall_ranking=None, save_path: str = None):
    """產生掃描報告 HTML"""
    
    html = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>全市場策略掃描報告</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00d4ff; margin-bottom: 20px; }
        h2 { color: #ff6b6b; margin: 30px 0 15px; font-size: 18px; }
        .meta { color: #888; margin-bottom: 30px; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 30px; background: #16213e; border-radius: 8px; overflow: hidden; }
        th { background: #0f3460; padding: 12px; text-align: left; color: #00d4ff; }
        td { padding: 10px 12px; border-bottom: 1px solid #0f3460; }
        tr:hover { background: #1f4068; }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .highlight { background: #2a3f5f; font-weight: bold; }
        .trophy { font-size: 1.5em; }
        .gold { color: #ffd700; }
        .silver { color: #c0c0c0; }
        .bronze { color: #cd7f32; }
    </style>
</head>
<body>
<div class="container">
    <h1>📊 全市場策略掃描報告</h1>
    <p class="meta">產生時間: """ + datetime.now().strftime('%Y-%m-%d %H:%M') + """ | 每策略顯示夏普比率 TOP 30</p>
"""
    
    # 加入總排名區塊（如果有）
    if overall_ranking is not None and not overall_ranking.empty:
        html += "\n<h2 class='trophy'>🏆 策略總排名 (TOP 30)</h2>\n"
        html += "<p style='color: #888; margin-bottom: 15px;'>綜合分數 = 出現策略數 × 平均夏普比率，能在越多策略中表現優異的股票排名越前</p>\n"
        html += "<table>\n<thead><tr>"
        html += "<th>排名</th><th>股票</th><th>名稱</th><th>綜合分數</th><th>策略數</th><th>平均夏普</th><th>平均報酬</th><th>最佳策略</th>"
        html += "</tr></thead>\n<tbody>\n"
        
        for rank, (_, row) in enumerate(overall_ranking.iterrows(), 1):
            # 前三名特殊標註
            if rank == 1:
                rank_str = '<span class="gold">🥇 1</span>'
            elif rank == 2:
                rank_str = '<span class="silver">🥈 2</span>'
            elif rank == 3:
                rank_str = '<span class="bronze">🥉 3</span>'
            else:
                rank_str = str(rank)
            
            ret_class = 'positive' if row['avg_return'] > 0 else 'negative'
            html += f"""<tr>
                <td>{rank_str}</td>
                <td><strong>{row['ticker']}</strong></td>
                <td>{row['name'][:8]}</td>
                <td><strong>{row['score']:.2f}</strong></td>
                <td>{row['strategy_count']}</td>
                <td>{row['avg_sharpe']:.2f}</td>
                <td class="{ret_class}">{row['avg_return']:.2%}</td>
                <td>{row['best_strategy']}</td>
            </tr>\n"""
        
        html += "</tbody></table>\n"
        html += "<hr style='border-color: #333; margin: 40px 0;'>\n"
    
    for strategy_name, df in results.items():
        if df.empty:
            continue
            
        html += f"\n<h2>🎯 {strategy_name}</h2>\n"
        html += "<table>\n<thead><tr>"
        html += "<th>排名</th><th>股票</th><th>名稱</th><th>報酬率</th><th>夏普比率</th><th>最大回撤</th><th>勝率</th><th>交易次數</th>"
        html += "</tr></thead>\n<tbody>\n"
        
        for i, row in df.head(30).iterrows():
            ret_class = 'positive' if row['total_return'] > 0 else 'negative'
            html += f"""<tr>
                <td>{df.index.get_loc(i) + 1}</td>
                <td><strong>{row['ticker']}</strong></td>
                <td>{row['name'][:8]}</td>
                <td class="{ret_class}">{row['total_return']:.2%}</td>
                <td><strong>{row['sharpe_ratio']:.2f}</strong></td>
                <td class="negative">{row['max_drawdown']:.2%}</td>
                <td>{row['win_rate']:.2%}</td>
                <td>{row['trade_count']}</td>
            </tr>\n"""
        
        html += "</tbody></table>\n"
    
    html += """
    <hr style="border-color: #333; margin: 40px 0;">
    <h2>📖 指標說明</h2>
    <table>
        <tr><td><strong>夏普比率 (Sharpe Ratio)</strong></td><td>風險調整後報酬。> 1 = 好，> 2 = 很好，> 3 = 優秀</td></tr>
        <tr><td><strong>總報酬率</strong></td><td>策略總獲利百分比</td></tr>
        <tr><td><strong>最大回撤</strong></td><td>最大虧損幅度（越小越好）</td></tr>
        <tr><td><strong>勝率</strong></td><td>獲利交易的比例</td></tr>
    </table>
</div>
</body>
</html>
"""
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"\n📄 報告已儲存: {save_path}")
    
    return html


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 全市場策略掃描工具")
    print("=" * 60)
    
    # 執行掃描
    results, overall_ranking = market_scan_all_strategies(top_n=30, min_volume=500)
    
    # 產生報告
    report_path = os.path.join(REPORT_DIR, "market_scan_all_strategies.html")
    generate_scan_report(results, overall_ranking=overall_ranking, save_path=report_path)
    
    print("\n✅ 掃描完成！")
    print(f"   報告位置: {report_path}")
