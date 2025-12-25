# -*- coding: utf-8 -*-
"""
視覺化報表模組
產生回測結果的圖表和 HTML 報告
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 嘗試導入 matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # 非互動模式
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ matplotlib 未安裝，部分圖表功能無法使用")


def plot_equity_curve(result: dict, title: str = None, save_path: str = None):
    """
    繪製權益曲線圖
    
    Args:
        result: BacktestEngine.run() 的回傳結果
        title: 圖表標題
        save_path: 儲存路徑（None 則顯示）
    """
    if not MATPLOTLIB_AVAILABLE:
        print("需要安裝 matplotlib: pip install matplotlib")
        return
    
    equity = result['equity_curve']
    metrics = result['metrics']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # 上圖：權益曲線
    ax1 = axes[0]
    ax1.plot(equity.values, linewidth=1.5, color='#2E86AB')
    ax1.axhline(y=metrics['initial_capital'], color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(range(len(equity)), metrics['initial_capital'], equity.values, 
                     where=equity.values >= metrics['initial_capital'], 
                     color='#28A745', alpha=0.3)
    ax1.fill_between(range(len(equity)), metrics['initial_capital'], equity.values,
                     where=equity.values < metrics['initial_capital'],
                     color='#DC3545', alpha=0.3)
    
    ax1.set_title(title or f"策略: {metrics.get('strategy', 'Unknown')}", fontsize=14, fontweight='bold')
    ax1.set_ylabel('權益 (NT$)')
    ax1.grid(True, alpha=0.3)
    
    # 加入績效資訊
    info_text = (f"報酬率: {metrics['total_return']:.2%}\n"
                 f"夏普比率: {metrics['sharpe_ratio']:.2f}\n"
                 f"最大回撤: {metrics['max_drawdown']:.2%}\n"
                 f"勝率: {metrics['win_rate']:.2%}")
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 下圖：回撤
    ax2 = axes[1]
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100
    ax2.fill_between(range(len(drawdown)), 0, drawdown.values, color='#DC3545', alpha=0.5)
    ax2.set_ylabel('回撤 (%)')
    ax2.set_xlabel('交易天數')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 圖表已儲存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_trades(result: dict, df: pd.DataFrame, save_path: str = None):
    """
    繪製股價走勢圖並標示買賣點
    
    Args:
        result: 回測結果
        df: 股價 DataFrame
        save_path: 儲存路徑
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    trades = result['trades']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 繪製股價
    ax.plot(df['close'].values, linewidth=1, color='#2E86AB', label='收盤價')
    
    # 標示買入點
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    for t in buy_trades:
        try:
            idx = df[df['date'].astype(str).str.contains(t['date'][:10])].index[0]
            ax.scatter(idx, t['price'], marker='^', color='#28A745', s=100, zorder=5)
        except:
            pass
    
    # 標示賣出點
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    for t in sell_trades:
        try:
            idx = df[df['date'].astype(str).str.contains(t['date'][:10])].index[0]
            ax.scatter(idx, t['price'], marker='v', color='#DC3545', s=100, zorder=5)
        except:
            pass
    
    ax.set_title(f"交易點位 - {result['metrics'].get('strategy', '')}", fontsize=12)
    ax.set_ylabel('價格')
    ax.set_xlabel('交易天數')
    ax.legend(['收盤價', '買入', '賣出'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def generate_html_report(result: dict, 
                         ticker: str = "Unknown",
                         save_path: str = None) -> str:
    """
    產生 HTML 格式的回測報告
    
    Args:
        result: 回測結果
        ticker: 股票代碼
        save_path: 儲存路徑
    
    Returns:
        str: HTML 內容
    """
    metrics = result['metrics']
    trades = result['trades']
    
    # 交易明細 HTML
    trades_html = ""
    for t in trades:
        if t['type'] == 'BUY':
            trades_html += f"""
            <tr class="buy">
                <td>{t['date']}</td>
                <td>🟢 買入</td>
                <td>${t['price']:,.2f}</td>
                <td>{t['shares']:,}</td>
                <td>-</td>
            </tr>"""
        else:
            profit_class = 'profit' if t.get('profit', 0) > 0 else 'loss'
            trades_html += f"""
            <tr class="sell">
                <td>{t['date']}</td>
                <td>🔴 賣出</td>
                <td>${t['price']:,.2f}</td>
                <td>{t['shares']:,}</td>
                <td class="{profit_class}">${t.get('profit', 0):,.0f} ({t.get('return', 0):.2%})</td>
            </tr>"""
    
    html = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>回測報告 - {ticker}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #2E86AB, #1a5276); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
        .header .subtitle {{ opacity: 0.8; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .card h2 {{ color: #2E86AB; margin-bottom: 15px; font-size: 18px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .metric {{ text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .metric .value {{ font-size: 24px; font-weight: bold; color: #2E86AB; }}
        .metric .label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .positive {{ color: #28A745 !important; }}
        .negative {{ color: #DC3545 !important; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .buy {{ background: #e8f5e9; }}
        .sell {{ background: #ffebee; }}
        .profit {{ color: #28A745; font-weight: bold; }}
        .loss {{ color: #DC3545; font-weight: bold; }}
        .footer {{ text-align: center; color: #999; margin-top: 30px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 回測報告</h1>
            <div class="subtitle">股票: {ticker} | 策略: {metrics.get('strategy', 'Unknown')} | 產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        
        <div class="card">
            <h2>績效摘要</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="value {'positive' if metrics['total_return'] > 0 else 'negative'}">{metrics['total_return']:.2%}</div>
                    <div class="label">總報酬率</div>
                </div>
                <div class="metric">
                    <div class="value">{metrics['sharpe_ratio']:.2f}</div>
                    <div class="label">夏普比率</div>
                </div>
                <div class="metric">
                    <div class="value negative">{metrics['max_drawdown']:.2%}</div>
                    <div class="label">最大回撤</div>
                </div>
                <div class="metric">
                    <div class="value">{metrics['win_rate']:.2%}</div>
                    <div class="label">勝率</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>資金變化</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="value">${metrics['initial_capital']:,.0f}</div>
                    <div class="label">初始資金</div>
                </div>
                <div class="metric">
                    <div class="value {'positive' if metrics['final_capital'] > metrics['initial_capital'] else 'negative'}">${metrics['final_capital']:,.0f}</div>
                    <div class="label">最終資金</div>
                </div>
                <div class="metric">
                    <div class="value">{metrics['trade_count']}</div>
                    <div class="label">交易次數</div>
                </div>
                <div class="metric">
                    <div class="value">{metrics['profit_factor']:.2f}</div>
                    <div class="label">盈虧比</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>交易明細</h2>
            <table>
                <thead>
                    <tr>
                        <th>日期</th>
                        <th>類型</th>
                        <th>價格</th>
                        <th>股數</th>
                        <th>損益</th>
                    </tr>
                </thead>
                <tbody>
                    {trades_html}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            由 tw-quant-simple 回測引擎產生
        </div>
    </div>
</body>
</html>
"""
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"📄 報告已儲存: {save_path}")
    
    return html


def print_summary(result: dict):
    """印出績效摘要"""
    metrics = result['metrics']
    
    print("\n" + "=" * 50)
    print(f"📊 {metrics.get('strategy', '策略')} 回測報告")
    print("=" * 50)
    
    print(f"\n💰 資金")
    print(f"   初始: ${metrics['initial_capital']:,.0f}")
    print(f"   最終: ${metrics['final_capital']:,.0f}")
    
    print(f"\n📈 報酬")
    ret_color = "🟢" if metrics['total_return'] > 0 else "🔴"
    print(f"   {ret_color} 總報酬: {metrics['total_return']:.2%}")
    print(f"   年化報酬: {metrics['annual_return']:.2%}")
    
    print(f"\n📉 風險")
    print(f"   夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"   最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"   年化波動: {metrics['volatility']:.2%}")
    
    print(f"\n🔄 交易")
    print(f"   交易次數: {metrics['trade_count']} 筆")
    print(f"   勝率: {metrics['win_rate']:.2%}")
    print(f"   盈虧比: {metrics['profit_factor']:.2f}")
    
    print("=" * 50)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from backtest import BacktestEngine, MACrossStrategy
    import pandas as pd
    
    # 測試
    df = pd.read_csv('data/tw-share/dayK/2330.TW_台積電.csv')
    engine = BacktestEngine()
    result = engine.run(df, MACrossStrategy(5, 20))
    
    # 印出摘要
    print_summary(result)
    
    # 產生 HTML 報告
    generate_html_report(result, ticker='2330.TW', save_path='reports/test_report.html')
