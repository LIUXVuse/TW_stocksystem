#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================
📊 策略訊號掃描器 v2.0
========================================

功能：
1. 🔥 今日大推個股（多策略共識 + 大成交量）
2. 策略智能排名（根據最近 30 天回測動態計算）
3. 各策略詳細訊號
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
from tqdm import tqdm
from collections import Counter

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
    ChipTechStrategy,
)

# 法人資料載入
try:
    from data_loader import load_institutional_data
    INSTITUTIONAL_DATA = load_institutional_data()
    HAS_INSTITUTIONAL = len(INSTITUTIONAL_DATA) > 0
    # 取得法人資料的日期範圍
    INSTITUTIONAL_DATES = sorted(INSTITUTIONAL_DATA.keys()) if INSTITUTIONAL_DATA else []
    INSTITUTIONAL_LATEST = INSTITUTIONAL_DATES[-1] if INSTITUTIONAL_DATES else None
    print(f"✅ 已載入法人資料: {len(INSTITUTIONAL_DATA)} 天 (最新: {INSTITUTIONAL_LATEST})")
except:
    INSTITUTIONAL_DATA = {}
    HAS_INSTITUTIONAL = False
    INSTITUTIONAL_DATES = []
    INSTITUTIONAL_LATEST = None
    print("⚠️ 無法載入法人資料，跳過法人策略")

# ========== 設定 ==========

# 成交量門檻（排除小型股）
MIN_VOLUME_THRESHOLD = 3000  # 日均成交量至少 3000 張
MIN_VOLUME_TOP_PICKS = 5000  # 今日大推需要 5000 張以上

# 市值前 N 大熱門股票（確保流動性）
POPULAR_TICKERS = [
    '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2303.TW',
    '2881.TW', '2882.TW', '2884.TW', '2886.TW', '2891.TW',
    '1301.TW', '1303.TW', '1326.TW', '2002.TW', '2412.TW',
    '3711.TW', '2357.TW', '3008.TW', '2382.TW', '2395.TW',
    '0050.TW', '0056.TW', '00878.TW', '00713.TW', '00919.TW',
]

# ========== 策略解說 ==========

STRATEGY_INFO = {
    'MA5x20': {
        'name': 'MA5x20 均線交叉',
        'desc': '5日線上穿20日線買入',
        'type': '趨勢',
        'risk': '中',
    },
    'MA5x60': {
        'name': 'MA5x60 中期均線',
        'desc': '5日線上穿60日線買入',
        'type': '趨勢',
        'risk': '低',
    },
    'MACD': {
        'name': 'MACD 金叉',
        'desc': 'MACD上穿信號線買入',
        'type': '動能',
        'risk': '中',
    },
    '動量突破': {
        'name': '動量突破',
        'desc': '突破20日高點+量增買入',
        'type': '突破',
        'risk': '高',
    },
    '海龜策略': {
        'name': '海龜策略',
        'desc': '突破20日高點買入',
        'type': '趨勢',
        'risk': '中',
    },
    '外資連買': {
        'name': '外資連續買超',
        'desc': '外資連續3天買超',
        'type': '籌碼',
        'risk': '低',
    },
    '投信連買': {
        'name': '投信連續買超',
        'desc': '投信連續3天買超',
        'type': '籌碼',
        'risk': '低',
    },
}


def calculate_dynamic_ranking(sample_size=50):
    """
    動態計算策略排名
    根據最近的實際回測結果
    包含：技術分析策略 + 法人籌碼策略
    """
    STOCK_DIR = 'data/tw-share/dayK'
    files = glob(os.path.join(STOCK_DIR, "*.csv"))
    
    # 取樣計算（加速）
    import random
    sample_files = random.sample(files, min(sample_size, len(files)))
    
    # 技術分析策略
    tech_strategies = [
        ("MA5x20", MACrossStrategy(5, 20)),
        ("MA5x60", MACrossStrategy(5, 60)),
        ("MACD", MACDStrategy()),
        ("動量突破", MomentumBreakoutStrategy(20)),
        ("海龜策略", TurtleStrategy(20, 10)),
    ]
    
    # 法人策略（如果有法人資料）
    inst_strategies = []
    if HAS_INSTITUTIONAL:
        inst_strategies = [
            ("外資連買", InstitutionalFollowStrategy('foreign', 3, threshold=50)),
            ("投信連買", InstitutionalFollowStrategy('trust', 3, threshold=5)),  # 極低門檻
        ]
    
    engine = BacktestEngine()
    strategy_scores = {name: [] for name, _ in tech_strategies + inst_strategies}
    
    for csv_path in sample_files:
        try:
            df = pd.read_csv(csv_path)
            if df['volume'].mean() < MIN_VOLUME_THRESHOLD:
                continue
            
            # 只用最近 60 天的資料計算
            df_recent = df.tail(60)
            ticker = os.path.basename(csv_path).split('_')[0]
            
            # 技術分析策略
            for name, strategy in tech_strategies:
                try:
                    result = engine.run(df_recent, strategy, verbose=False)
                    if result['metrics']['trade_count'] > 0:
                        strategy_scores[name].append(result['metrics']['sharpe_ratio'])
                except:
                    continue
            
            # 法人策略（需要合併法人資料）
            if HAS_INSTITUTIONAL and INSTITUTIONAL_LATEST:
                try:
                    df_full = df.copy()
                    df_full['date_str'] = pd.to_datetime(df_full['date']).dt.strftime('%Y%m%d')
                    df_with_inst = df_full[df_full['date_str'] <= INSTITUTIONAL_LATEST].copy()
                    
                    if len(df_with_inst) < 30:
                        continue
                    
                    # 初始化法人欄位
                    df_with_inst['foreign'] = 0
                    df_with_inst['trust'] = 0
                    df_with_inst['dealer'] = 0
                    df_with_inst['inst_total'] = 0
                    
                    # 填入法人資料
                    for i, row in df_with_inst.iterrows():
                        date_str = row['date_str']
                        if date_str in INSTITUTIONAL_DATA:
                            stock_data = INSTITUTIONAL_DATA[date_str].get(ticker, {})
                            df_with_inst.at[i, 'foreign'] = stock_data.get('foreign', 0)
                            df_with_inst.at[i, 'trust'] = stock_data.get('trust', 0)
                            df_with_inst.at[i, 'dealer'] = stock_data.get('dealer', 0)
                            df_with_inst.at[i, 'inst_total'] = stock_data.get('total', 0)
                    
                    # 只用最近 60 天
                    df_inst_recent = df_with_inst.tail(60)
                    
                    for name, strategy in inst_strategies:
                        try:
                            result = engine.run(df_inst_recent, strategy, verbose=False)
                            if result['metrics']['trade_count'] > 0:
                                strategy_scores[name].append(result['metrics']['sharpe_ratio'])
                        except:
                            continue
                except:
                    continue
        except:
            continue
    
    # 計算平均夏普
    rankings = []
    for name, scores in strategy_scores.items():
        if scores:
            avg_sharpe = np.mean(scores)
            rankings.append({
                'strategy': name,
                'avg_sharpe': avg_sharpe,
                'sample_count': len(scores),
                'info': STRATEGY_INFO.get(name, {})
            })
    
    # 按夏普排序
    rankings.sort(key=lambda x: x['avg_sharpe'], reverse=True)
    
    # 加入排名
    for i, r in enumerate(rankings):
        r['rank'] = i + 1
        if r['avg_sharpe'] >= 1.5:
            r['recommendation'] = '⭐⭐⭐ 強烈推薦'
        elif r['avg_sharpe'] >= 1.0:
            r['recommendation'] = '⭐⭐ 推薦'
        elif r['avg_sharpe'] >= 0.5:
            r['recommendation'] = '⭐ 可用'
        else:
            r['recommendation'] = '謹慎使用'
    
    return rankings


def scan_recent_signals(days=5):
    """
    掃描最近 N 天出現買入訊號的股票
    回傳所有訊號 + 成交量資訊
    """
    STOCK_DIR = 'data/tw-share/dayK'
    files = glob(os.path.join(STOCK_DIR, "*.csv"))
    
    # 技術分析策略
    strategies = [
        ("MA5x20", MACrossStrategy(5, 20)),
        ("MA5x60", MACrossStrategy(5, 60)),
        ("MACD", MACDStrategy()),
        ("動量突破", MomentumBreakoutStrategy(20)),
        ("海龜策略", TurtleStrategy(20, 10)),
    ]
    
    # 法人策略（需要法人資料）
    institutional_strategies = [
        ("外資連買", InstitutionalFollowStrategy('foreign', 3)),
        ("投信連買", InstitutionalFollowStrategy('trust', 3)),
    ]
    
    signals_found = []
    
    print(f"\n🔍 掃描最近 {days} 個交易日的買入訊號...")
    print(f"   股票數: {len(files)} 檔")
    print(f"   成交量門檻: {MIN_VOLUME_THRESHOLD} 張/日")
    print()
    
    for csv_path in tqdm(files, desc="掃描中"):
        try:
            df = pd.read_csv(csv_path)
            
            # 計算平均成交量
            avg_volume = df['volume'].mean()
            
            # 過濾成交量太低的
            if avg_volume < MIN_VOLUME_THRESHOLD:
                continue
            
            ticker = os.path.basename(csv_path).split('_')[0]
            name = os.path.basename(csv_path).replace('.csv', '').split('_', 1)[-1]
            
            # 取最後幾天
            recent = df.tail(days + 60)
            
            for strategy_name, strategy in strategies:
                try:
                    signals = strategy.generate_signals(recent)
                    last_n_signals = signals.tail(days)
                    
                    for i, (idx, sig) in enumerate(last_n_signals.items()):
                        if sig == 1:
                            signal_date = recent.loc[idx, 'date'] if 'date' in recent.columns else str(idx)
                            price = recent.loc[idx, 'close']
                            
                            signals_found.append({
                                'ticker': ticker,
                                'name': name[:10],
                                'strategy': strategy_name,
                                'signal_date': str(signal_date)[:10],
                                'price': price,
                                'days_ago': days - i - 1,
                                'avg_volume': avg_volume,
                                'is_popular': ticker in POPULAR_TICKERS,
                            })
                except:
                    continue
            
            # ===== 法人策略掃描 =====
            if HAS_INSTITUTIONAL and INSTITUTIONAL_LATEST:
                # 合併法人資料到 DataFrame
                # 使用完整資料而非 tail，因為法人資料可能只有到較早的日期
                df_full = df.copy()
                df_full['date_str'] = pd.to_datetime(df_full['date']).dt.strftime('%Y%m%d')
                
                # 只保留有法人資料的日期範圍
                df_with_inst = df_full[df_full['date_str'] <= INSTITUTIONAL_LATEST].copy()
                
                if len(df_with_inst) < 10:
                    continue  # 資料太少，跳過
                
                # 初始化法人欄位
                df_with_inst['foreign'] = 0
                df_with_inst['trust'] = 0
                df_with_inst['dealer'] = 0
                df_with_inst['inst_total'] = 0
                
                # 填入法人資料
                for i, row in df_with_inst.iterrows():
                    date_str = row['date_str']
                    if date_str in INSTITUTIONAL_DATA:
                        stock_data = INSTITUTIONAL_DATA[date_str].get(ticker, {})
                        df_with_inst.at[i, 'foreign'] = stock_data.get('foreign', 0)
                        df_with_inst.at[i, 'trust'] = stock_data.get('trust', 0)
                        df_with_inst.at[i, 'dealer'] = stock_data.get('dealer', 0)
                        df_with_inst.at[i, 'inst_total'] = stock_data.get('total', 0)
                
                # 用法人策略掃描（只用有法人資料的部分）
                for strategy_name, strategy in institutional_strategies:
                    try:
                        # 取法人資料範圍內的最後 N 天
                        recent_inst = df_with_inst.tail(days + 60)
                        signals = strategy.generate_signals(recent_inst)
                        last_n_signals = signals.tail(days)
                        
                        for i, (idx, sig) in enumerate(last_n_signals.items()):
                            if sig == 1:
                                signal_date = df_with_inst.loc[idx, 'date'] if 'date' in df_with_inst.columns else str(idx)
                                price = df_with_inst.loc[idx, 'close']
                                
                                signals_found.append({
                                    'ticker': ticker,
                                    'name': name[:10],
                                    'strategy': strategy_name,
                                    'signal_date': str(signal_date)[:10],
                                    'price': price,
                                    'days_ago': days - i - 1,
                                    'avg_volume': avg_volume,
                                    'is_popular': ticker in POPULAR_TICKERS,
                                })
                    except:
                        continue
                    
        except:
            continue
    
    return pd.DataFrame(signals_found)


def get_top_picks(signals_df):
    """
    找出今日大推個股
    條件：
    1. 成交量大（> MIN_VOLUME_TOP_PICKS）
    2. 被多個策略同時看好
    3. 訊號越新越好（今天或昨天）
    """
    if signals_df.empty:
        return pd.DataFrame()
    
    # 過濾高成交量
    high_vol = signals_df[signals_df['avg_volume'] >= MIN_VOLUME_TOP_PICKS]
    
    # 只看最近 2 天的訊號
    recent_signals = high_vol[high_vol['days_ago'] <= 1]
    
    if recent_signals.empty:
        # 如果沒有，放寬到 3 天
        recent_signals = high_vol[high_vol['days_ago'] <= 2]
    
    if recent_signals.empty:
        return pd.DataFrame()
    
    # 計算每支股票被幾個策略看好
    ticker_strategies = recent_signals.groupby('ticker').agg({
        'strategy': lambda x: list(set(x)),
        'name': 'first',
        'price': 'last',
        'signal_date': 'last',
        'avg_volume': 'first',
        'days_ago': 'min',
        'is_popular': 'first',
    }).reset_index()
    
    ticker_strategies['strategy_count'] = ticker_strategies['strategy'].apply(len)
    ticker_strategies['strategies_str'] = ticker_strategies['strategy'].apply(lambda x: ', '.join(x))
    
    # 優先排序：策略數量 > 熱門股 > 成交量
    ticker_strategies = ticker_strategies.sort_values(
        ['strategy_count', 'is_popular', 'avg_volume'], 
        ascending=[False, False, False]
    )
    
    return ticker_strategies.head(15)


def generate_signal_report_v2(signals_df, rankings, save_path='reports/signal_alert.html'):
    """產生訊號提醒報告 v2.0"""
    
    top_picks = get_top_picks(signals_df)
    
    # 建立策略評分對照
    strategy_scores = {r['strategy']: r for r in rankings}
    
    html = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>📢 訊號提醒 - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #58a6ff; margin-bottom: 10px; }}
        h2 {{ color: #f0883e; margin: 30px 0 15px; font-size: 20px; }}
        .meta {{ color: #8b949e; margin-bottom: 30px; }}
        
        .top-picks {{ background: linear-gradient(135deg, #1a3a1a 0%, #1a2a1a 100%); padding: 25px; border-radius: 12px; margin-bottom: 30px; border: 2px solid #238636; }}
        .top-picks h2 {{ color: #3fb950; margin-bottom: 20px; font-size: 24px; }}
        .top-picks-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 15px; }}
        .pick-card {{ background: #161b22; padding: 15px; border-radius: 8px; border-left: 4px solid #238636; }}
        .pick-card .ticker {{ font-size: 20px; font-weight: bold; color: #58a6ff; }}
        .pick-card .name {{ color: #8b949e; margin-bottom: 8px; }}
        .pick-card .strategies {{ color: #3fb950; font-size: 14px; }}
        .pick-card .price {{ color: #f0883e; margin-top: 8px; }}
        .pick-card .volume {{ color: #8b949e; font-size: 12px; }}
        
        .ranking {{ background: #161b22; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .ranking h3 {{ color: #f0883e; margin-bottom: 15px; }}
        .ranking-note {{ color: #8b949e; font-size: 12px; margin-bottom: 15px; font-style: italic; }}
        
        table {{ width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; margin-bottom: 20px; }}
        th {{ background: #21262d; padding: 12px; text-align: left; color: #58a6ff; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #21262d; }}
        tr:hover {{ background: #1f2937; }}
        .today {{ background: #1f3d1f !important; }}
        .star {{ color: #f0883e; }}
        
        .alert {{ background: #2d1f1f; padding: 15px; border-radius: 8px; border-left: 4px solid #f85149; margin-top: 30px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📢 訊號提醒</h1>
    <p class="meta">產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 成交量門檻: {MIN_VOLUME_TOP_PICKS//1000}K 張/日</p>
"""
    
    # ===== 今日大推區塊 =====
    if not top_picks.empty:
        html += """
    <div class="top-picks">
        <h2>🔥 今日大推個股</h2>
        <p style="color: #8b949e; margin-bottom: 15px;">被多個策略同時看好 + 成交量充足的股票</p>
        <div class="top-picks-grid">
"""
        for _, row in top_picks.iterrows():
            badge = "🏆" if row['strategy_count'] >= 3 else "⭐" if row['strategy_count'] >= 2 else ""
            days_text = "今天" if row['days_ago'] == 0 else f"{row['days_ago']} 天前"
            pop_badge = "💎" if row['is_popular'] else ""
            
            html += f"""
            <div class="pick-card">
                <div class="ticker">{badge} {row['ticker']} {pop_badge}</div>
                <div class="name">{row['name']}</div>
                <div class="strategies">✅ {row['strategy_count']} 個策略看好: {row['strategies_str']}</div>
                <div class="price">💰 ${row['price']:,.2f} ({days_text})</div>
                <div class="volume">📊 日均量 {row['avg_volume']/1000:.1f}K 張</div>
            </div>
"""
        html += """
        </div>
    </div>
"""
    else:
        html += """
    <div class="top-picks" style="border-color: #f0883e;">
        <h2>🔥 今日大推個股</h2>
        <p style="color: #f0883e;">目前沒有符合條件的大推個股（需要多策略共識 + 高成交量）</p>
    </div>
"""
    
    # ===== 策略排名 =====
    html += """
    <div class="ranking">
        <h3>📊 策略效果排名</h3>
        <p class="ranking-note">⚡ 此排名根據最近 60 天回測動態計算，每次掃描會更新</p>
        <table>
            <tr><th>排名</th><th>策略</th><th>夏普比率</th><th>評價</th><th>策略類型</th></tr>
"""
    
    for r in rankings:
        info = r.get('info', {})
        html += f"""
            <tr>
                <td>#{r['rank']}</td>
                <td><strong>{r['strategy']}</strong></td>
                <td>{r['avg_sharpe']:.2f}</td>
                <td class="star">{r['recommendation']}</td>
                <td>{info.get('type', '')}</td>
            </tr>
"""
    
    html += """
        </table>
    </div>
"""
    
    # ===== 各策略訊號 =====
    signals_df = signals_df[signals_df['avg_volume'] >= MIN_VOLUME_THRESHOLD]
    
    # 加入策略排名
    signals_df['rank'] = signals_df['strategy'].map(
        lambda x: strategy_scores.get(x, {}).get('rank', 99)
    )
    signals_df = signals_df.sort_values(['rank', 'avg_volume'], ascending=[True, False])
    
    for strategy_name in signals_df['strategy'].unique():
        strategy_signals = signals_df[signals_df['strategy'] == strategy_name]
        rank_info = strategy_scores.get(strategy_name, {})
        recommendation = rank_info.get('recommendation', '')
        
        html += f"""
    <h2>📈 {strategy_name} <span class="star">{recommendation}</span></h2>
    <table>
        <tr><th>股票</th><th>名稱</th><th>訊號日期</th><th>幾天前</th><th>價格</th><th>日均量(K)</th></tr>
"""
        
        # 按成交量排序，只顯示前 15 筆
        for _, row in strategy_signals.head(15).iterrows():
            row_class = 'today' if row['days_ago'] == 0 else ''
            days_text = '今天' if row['days_ago'] == 0 else f"{row['days_ago']} 天前"
            pop_badge = " 💎" if row['is_popular'] else ""
            
            html += f"""
        <tr class="{row_class}">
            <td><strong>{row['ticker']}</strong>{pop_badge}</td>
            <td>{row['name']}</td>
            <td>{row['signal_date']}</td>
            <td>{days_text}</td>
            <td>${row['price']:,.2f}</td>
            <td>{row['avg_volume']/1000:.1f}</td>
        </tr>
"""
        
        html += "</table>\n"
    
    html += """
    <div class="alert">
        <strong>⚠️ 風險提醒</strong><br>
        訊號僅供參考，不構成投資建議。過去績效不保證未來報酬。請自行評估風險。<br><br>
        💎 = 熱門大型股 | 🏆 = 3+ 策略共識 | ⭐ = 2 策略共識
    </div>
</div>
</body>
</html>
"""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return save_path


def main():
    print("\n" + "=" * 60)
    print("📊 策略訊號掃描器 v2.0")
    print("=" * 60)
    
    print("\n【1】動態計算策略排名...")
    print("   (根據最近 60 天回測結果)")
    rankings = calculate_dynamic_ranking(sample_size=100)
    
    print(f"\n{'排名':<6} {'策略':<12} {'夏普':<10} {'評價'}")
    print("-" * 50)
    for r in rankings:
        print(f"#{r['rank']:<5} {r['strategy']:<12} {r['avg_sharpe']:<10.2f} {r['recommendation']}")
    
    print("\n【2】掃描訊號...")
    signals = scan_recent_signals(days=5)
    
    if not signals.empty:
        print("\n【3】產生報告...")
        report_path = generate_signal_report_v2(signals, rankings)
        
        print(f"\n✅ 完成！")
        print(f"📄 報告: {report_path}")
        
        # 顯示今日大推
        top_picks = get_top_picks(signals)
        if not top_picks.empty:
            print("\n🔥 今日大推:")
            for _, row in top_picks.head(5).iterrows():
                print(f"   {row['ticker']} {row['name']} - {row['strategy_count']} 個策略看好")
    else:
        print("\n❌ 沒有找到訊號")


if __name__ == '__main__':
    main()
