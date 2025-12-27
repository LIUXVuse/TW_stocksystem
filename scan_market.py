# -*- coding: utf-8 -*-
"""
========================================
📊 全市場策略掃描工具 (夏普比率掃描) v2.0
========================================

功能：
1. 掃描全市場股票，用 10 種策略回測
2. 根據「夏普比率」排名，找出適合各策略的股票
3. 產生「跨策略總排名」綜合推薦股票

v2.0 優化項目：
- ✅ 多進程並行處理（加速 3-4 倍）
- ✅ 進度顯示與預估時間
- ✅ 智慧篩選（跳過不活躍股票）
- ✅ 分批處理，可中斷恢復
- ✅ 防止重複執行

用法：
    python scan_market.py           # 完整掃描
    python scan_market.py --fast    # 快速模式（只掃描活躍股票）
    python scan_market.py --resume  # 從上次中斷處繼續

報告輸出：
    reports/market_scan_all_strategies.html
"""
import sys
import os
import json
import time
import signal
import filelock
from multiprocessing import Pool, cpu_count
from functools import partial

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

# 資料目錄
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOCK_DIR = os.path.join(BASE_DIR, "data", "tw-share", "dayK")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
LOCK_FILE = os.path.join(CACHE_DIR, "scan_market.lock")
PROGRESS_FILE = os.path.join(CACHE_DIR, "scan_progress.json")

# 確保目錄存在
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def get_all_strategies(include_institutional=True, institutional_data=None):
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
    
    if include_institutional and institutional_data is not None:
        strategies.extend([
            ("外資連買", InstitutionalFollowStrategy('foreign', 3, threshold=100)),
            ("投信連買", InstitutionalFollowStrategy('trust', 3, threshold=10)),
        ])
    
    return strategies


def process_single_stock(args):
    """
    處理單一股票的回測（供多進程呼叫）
    
    Returns:
        dict: {strategy_name: [result_dict, ...]}
    """
    csv_path, strategy_configs, min_volume, min_days = args
    
    try:
        # 讀取股價資料
        df = pd.read_csv(csv_path)
        
        # 基本過濾
        if len(df) < min_days:
            return None
        
        if df['volume'].mean() < min_volume:
            return None
        
        # 股票資訊
        ticker = os.path.basename(csv_path).split('_')[0]
        name = os.path.basename(csv_path).replace('.csv', '').split('_', 1)[-1]
        
        # 嘗試載入法人資料
        df_with_inst = None
        try:
            df_with_inst = load_stock_with_institutional(ticker)
        except:
            pass
        
        # 初始化回測引擎
        engine = BacktestEngine()
        stock_results = {}
        
        # 執行各策略回測
        for strategy_name, strategy_type, strategy_params in strategy_configs:
            try:
                # 重建策略實例（因為多進程不能序列化策略物件）
                strategy = create_strategy(strategy_type, strategy_params)
                
                # 選擇資料
                if '連買' in strategy_name or '連賣' in strategy_name:
                    if df_with_inst is None or df_with_inst.empty:
                        continue
                    run_df = df_with_inst
                else:
                    run_df = df
                
                # 執行回測
                result = engine.run(run_df, strategy, verbose=False)
                m = result['metrics']
                
                # 篩選有效結果
                if m['trade_count'] >= 3:
                    if strategy_name not in stock_results:
                        stock_results[strategy_name] = []
                    
                    stock_results[strategy_name].append({
                        'ticker': ticker,
                        'name': name,
                        'total_return': m['total_return'],
                        'sharpe_ratio': m['sharpe_ratio'],
                        'max_drawdown': m['max_drawdown'],
                        'win_rate': m['win_rate'],
                        'trade_count': m['trade_count']
                    })
            except Exception:
                continue
        
        return stock_results if stock_results else None
        
    except Exception:
        return None


def create_strategy(strategy_type, params):
    """根據類型和參數建立策略實例"""
    strategy_map = {
        'MACross': MACrossStrategy,
        'RSI': RSIStrategy,
        'MACD': MACDStrategy,
        'Bollinger': BollingerStrategy,
        'Momentum': MomentumBreakoutStrategy,
        'Volume': VolumeBreakoutStrategy,
        'Turtle': TurtleStrategy,
        'Institutional': InstitutionalFollowStrategy,
    }
    return strategy_map[strategy_type](*params)


def get_strategy_configs(include_institutional=True, institutional_data=None):
    """取得策略配置（可序列化版本）"""
    configs = [
        ("MA5x20", "MACross", (5, 20)),
        ("MA5x60", "MACross", (5, 60)),
        ("RSI", "RSI", (30, 70)),
        ("MACD", "MACD", ()),
        ("布林通道", "Bollinger", ()),
        ("動量突破", "Momentum", (20,)),
        ("量價突破", "Volume", (2.0,)),
        ("海龜策略", "Turtle", (20, 10)),
    ]
    
    if include_institutional and institutional_data is not None:
        configs.extend([
            ("外資連買", "Institutional", ('foreign', 3, 100)),
            ("投信連買", "Institutional", ('trust', 3, 10)),
        ])
    
    return configs


def compute_overall_ranking(results: dict, top_n=30):
    """計算跨策略總排名"""
    stock_stats = {}
    
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
    
    ranking_df = pd.DataFrame(ranking_data)
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values('score', ascending=False).head(top_n)
    
    return ranking_df


def save_progress(processed_files, results, start_time):
    """儲存處理進度"""
    progress = {
        'processed_files': processed_files,
        'results': {k: v.to_dict('records') if isinstance(v, pd.DataFrame) else v for k, v in results.items()},
        'start_time': start_time,
        'save_time': time.time()
    }
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False)


def load_progress():
    """載入處理進度"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            # 檢查進度是否過期（超過 1 天）
            if time.time() - progress.get('save_time', 0) > 86400:
                return None
            return progress
        except:
            return None
    return None


def clear_progress():
    """清除進度檔案"""
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


def market_scan_all_strategies(top_n=30, min_volume=500, min_days=60, 
                                fast_mode=False, resume=False, num_workers=None):
    """
    全市場掃描所有策略（多進程優化版）
    
    Args:
        top_n: 每個策略取前 N 名
        min_volume: 最低平均成交量（萬股），低於此值跳過
        min_days: 最低數據天數
        fast_mode: 快速模式（提高成交量門檻）
        resume: 從上次中斷處繼續
        num_workers: 並行工作數（預設為 CPU 核心數）
    """
    # 載入法人資料
    institutional_data = None
    try:
        institutional_data = load_institutional_data()
        print(f"✅ 已載入法人資料: {len(institutional_data)} 天")
    except:
        print("⚠️ 無法載入法人資料，法人策略將跳過")
    
    # 取得所有股票檔案
    all_files = glob(os.path.join(STOCK_DIR, "*.csv"))
    
    # 快速模式提高門檻
    if fast_mode:
        min_volume = max(min_volume, 2000)
        print("⚡ 快速模式：只掃描高成交量股票")
    
    # 取得策略配置
    strategy_configs = get_strategy_configs(True, institutional_data)
    
    # 初始化結果
    results = {name: [] for name, _, _ in strategy_configs}
    processed_files = set()
    
    # 嘗試恢復進度
    if resume:
        progress = load_progress()
        if progress:
            processed_files = set(progress['processed_files'])
            for name, data in progress['results'].items():
                if name in results and data:
                    results[name] = data
            print(f"📂 從上次進度恢復，已處理 {len(processed_files)} 檔")
    
    # 過濾已處理的檔案
    files_to_process = [f for f in all_files if f not in processed_files]
    
    # 統計資訊
    total_strategies = len(strategy_configs)
    total_combinations = len(files_to_process) * total_strategies
    
    print(f"\n🔍 全市場掃描（v2.0 優化版）")
    print(f"   股票數: {len(all_files)} 檔（待處理: {len(files_to_process)} 檔）")
    print(f"   策略數: {total_strategies} 種")
    print(f"   總組合: {total_combinations:,} 次回測")
    print(f"   最低成交量: {min_volume:,} 股")
    print()
    
    # 決定工作進程數
    if num_workers is None:
        num_workers = min(cpu_count(), 6)  # 最多用 6 核心
    
    print(f"🚀 使用 {num_workers} 個進程並行處理...")
    print()
    
    # 準備任務參數
    tasks = [(f, strategy_configs, min_volume, min_days) for f in files_to_process]
    
    # 開始時間
    start_time = time.time()
    processed_count = 0
    
    # 使用多進程處理
    try:
        with Pool(processes=num_workers) as pool:
            # 使用 imap_unordered 以便即時更新進度
            for i, stock_result in enumerate(tqdm(
                pool.imap_unordered(process_single_stock, tasks),
                total=len(tasks),
                desc="掃描中",
                unit="檔"
            )):
                if stock_result:
                    for strategy_name, strategy_results in stock_result.items():
                        results[strategy_name].extend(strategy_results)
                
                processed_count += 1
                processed_files.add(files_to_process[i] if i < len(files_to_process) else "")
                
                # 每 100 檔儲存一次進度
                if processed_count % 100 == 0:
                    save_progress(list(processed_files), results, start_time)
                    
                    # 顯示預估時間
                    elapsed = time.time() - start_time
                    if processed_count > 0:
                        eta = (elapsed / processed_count) * (len(tasks) - processed_count)
                        tqdm.write(f"   ⏱️ 已用時間: {elapsed/60:.1f} 分鐘 | 預計剩餘: {eta/60:.1f} 分鐘")
                        
    except KeyboardInterrupt:
        print("\n\n⚠️ 使用者中斷，儲存進度...")
        save_progress(list(processed_files), results, start_time)
        print("   進度已儲存，下次使用 --resume 繼續")
        raise
    
    # 計算總時間
    total_time = time.time() - start_time
    print(f"\n⏱️ 總耗時: {total_time/60:.1f} 分鐘")
    
    # 轉換為 DataFrame 並排序
    for name in results:
        if results[name]:
            df = pd.DataFrame(results[name])
            results[name] = df.sort_values('sharpe_ratio', ascending=False).head(top_n)
        else:
            results[name] = pd.DataFrame()
    
    # 計算跨策略總排名
    overall_ranking = compute_overall_ranking(results)
    
    # 清除進度檔案（成功完成）
    clear_progress()
    
    return results, overall_ranking


def generate_scan_report(results: dict, overall_ranking=None, save_path: str = None, scan_time=None):
    """產生掃描報告 HTML"""
    
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    scan_info = f" | 掃描耗時: {scan_time:.1f} 分鐘" if scan_time else ""
    
    html = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>全市場策略掃描報告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #00d4ff; margin-bottom: 20px; }}
        h2 {{ color: #ff6b6b; margin: 30px 0 15px; font-size: 18px; }}
        .meta {{ color: #888; margin-bottom: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; background: #16213e; border-radius: 8px; overflow: hidden; }}
        th {{ background: #0f3460; padding: 12px; text-align: left; color: #00d4ff; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #0f3460; }}
        tr:hover {{ background: #1f4068; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .highlight {{ background: #2a3f5f; font-weight: bold; }}
        .trophy {{ font-size: 1.5em; }}
        .gold {{ color: #ffd700; }}
        .silver {{ color: #c0c0c0; }}
        .bronze {{ color: #cd7f32; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 全市場策略掃描報告</h1>
    <p class="meta">產生時間: {time_str}{scan_info} | 每策略顯示夏普比率 TOP 30</p>
"""
    
    # 加入總排名區塊
    if overall_ranking is not None and not overall_ranking.empty:
        html += "\n<h2 class='trophy'>🏆 策略總排名 (TOP 30)</h2>\n"
        html += "<p style='color: #888; margin-bottom: 15px;'>綜合分數 = 出現策略數 × 平均夏普比率，能在越多策略中表現優異的股票排名越前</p>\n"
        html += "<table>\n<thead><tr>"
        html += "<th>排名</th><th>股票</th><th>名稱</th><th>綜合分數</th><th>策略數</th><th>平均夏普</th><th>平均報酬</th><th>最佳策略</th>"
        html += "</tr></thead>\n<tbody>\n"
        
        for rank, (_, row) in enumerate(overall_ranking.iterrows(), 1):
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
        if isinstance(df, pd.DataFrame) and df.empty:
            continue
        if isinstance(df, list) and not df:
            continue
            
        html += f"\n<h2>🎯 {strategy_name}</h2>\n"
        html += "<table>\n<thead><tr>"
        html += "<th>排名</th><th>股票</th><th>名稱</th><th>報酬率</th><th>夏普比率</th><th>最大回撤</th><th>勝率</th><th>交易次數</th>"
        html += "</tr></thead>\n<tbody>\n"
        
        if isinstance(df, pd.DataFrame):
            for idx, (_, row) in enumerate(df.head(30).iterrows(), 1):
                ret_class = 'positive' if row['total_return'] > 0 else 'negative'
                html += f"""<tr>
                    <td>{idx}</td>
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
    import argparse
    
    parser = argparse.ArgumentParser(description='全市場策略掃描工具')
    parser.add_argument('--fast', action='store_true', help='快速模式（只掃描高成交量股票）')
    parser.add_argument('--resume', action='store_true', help='從上次中斷處繼續')
    parser.add_argument('--workers', type=int, default=None, help='並行工作數')
    parser.add_argument('--min-volume', type=int, default=500, help='最低成交量（預設 500）')
    args = parser.parse_args()
    
    # 使用檔案鎖防止重複執行
    try:
        lock = filelock.FileLock(LOCK_FILE, timeout=1)
        with lock:
            print("=" * 60)
            print("🚀 全市場策略掃描工具 v2.0")
            print("=" * 60)
            
            start_time = time.time()
            
            # 執行掃描
            results, overall_ranking = market_scan_all_strategies(
                top_n=30,
                min_volume=args.min_volume,
                fast_mode=args.fast,
                resume=args.resume,
                num_workers=args.workers
            )
            
            scan_time = (time.time() - start_time) / 60
            
            # 產生報告
            report_path = os.path.join(REPORT_DIR, "market_scan_all_strategies.html")
            generate_scan_report(results, overall_ranking=overall_ranking, 
                               save_path=report_path, scan_time=scan_time)
            
            print("\n✅ 掃描完成！")
            print(f"   報告位置: {report_path}")
            
    except filelock.Timeout:
        print("❌ 錯誤：已有另一個掃描程序正在執行！")
        print("   如確定沒有，請刪除: " + LOCK_FILE)
        sys.exit(1)
