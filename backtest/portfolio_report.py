# -*- coding: utf-8 -*-
"""
投資組合回測報表模組
產生多股投組的 HTML 報表與 CSV 匯出
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

def generate_portfolio_html_report(result: dict, 
                                    tickers: list = None,
                                    strategy_name: str = "Portfolio Strategy",
                                    save_path: str = None) -> str:
    """
    產生投資組合回測的 HTML 報告
    
    Args:
        result: PortfolioEngine.run() 的回傳結果
        tickers: 股票代碼列表
        strategy_name: 策略名稱
        save_path: 儲存路徑
        
    Returns:
        str: HTML 內容
    """
    metrics = result.get('metrics', {})
    trades_df = result.get('trades', pd.DataFrame())
    positions = result.get('positions', {})
    equity_curve = result.get('equity_curve', pd.Series())
    
    # 基本資訊
    ticker_str = ', '.join(tickers) if tickers else 'N/A'
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # 計算報酬顏色
    total_return = metrics.get('total_return', 0)
    return_class = 'positive' if total_return >= 0 else 'negative'
    
    # 盈虧比顯示（處理 inf 和 0）
    profit_factor = metrics.get('profit_factor', 0)
    if profit_factor == float('inf') or profit_factor > 9999:
        profit_factor_str = '🏆 全勝'
    elif profit_factor == 0 or profit_factor is None:
        profit_factor_str = 'N/A'
    else:
        profit_factor_str = f'{profit_factor:.2f}'
    
    # 計算最大回撤顏色
    max_dd = metrics.get('max_drawdown', 0)
    dd_class = 'negative'
    
    # 權益曲線資料 (轉成 JSON 用於圖表)
    if not equity_curve.empty:
        chart_dates = equity_curve.index.strftime('%Y-%m-%d').tolist()
        chart_values = equity_curve.values.tolist()
    else:
        chart_dates = []
        chart_values = []
    
    # 計算資產配置 (最終持倉)
    position_items = []
    total_value = sum(v * 100 for v in positions.values() if v > 0)  # 假設價格，這裡先簡化
    for ticker, shares in positions.items():
        if shares > 0:
            position_items.append(f'<div class="position-item"><span class="ticker">{ticker}</span><span class="shares">{shares:,} 股</span></div>')
    positions_html = '\n'.join(position_items) if position_items else '<p>無持倉</p>'
    
    # 交易明細 (全部顯示，最新的在前面)
    trade_rows = []
    if not trades_df.empty:
        # 反轉順序，最新的在前面
        all_trades = trades_df.iloc[::-1]
        for _, t in all_trades.iterrows():
            trade_type = t.get('type', 'N/A')
            type_class = 'buy' if trade_type == 'BUY' else 'sell'
            type_icon = '🟢 買入' if trade_type == 'BUY' else '🔴 賣出'
            profit = t.get('profit', 0)
            profit_str = '-'
            profit_class = ''
            if trade_type == 'SELL' and profit != 0:
                profit_class = 'profit' if profit > 0 else 'loss'
                profit_str = f'${profit:,.0f}'
            reason = t.get('reason', '')  # 進出場原因
            trade_rows.append(f'''
            <tr class="{type_class}">
                <td>{t.get('date', 'N/A')}</td>
                <td>{t.get('ticker', 'N/A')}</td>
                <td>{type_icon}</td>
                <td>${t.get('price', 0):,.2f}</td>
                <td>{t.get('shares', 0):,}</td>
                <td class="{profit_class}">{profit_str}</td>
                <td>{reason}</td>
            </tr>''')
    trades_html = '\n'.join(trade_rows) if trade_rows else '<tr><td colspan="7">無交易記錄</td></tr>'
    
    html = f'''
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>投資組合回測報告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #238636, #1a5276); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
        .header .subtitle {{ opacity: 0.8; font-size: 14px; }}
        .card {{ background: #161b22; border-radius: 10px; padding: 20px; margin-bottom: 20px; border: 1px solid #30363d; }}
        .card h2 {{ color: #58a6ff; margin-bottom: 15px; font-size: 18px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .metric {{ text-align: center; padding: 15px; background: #21262d; border-radius: 8px; }}
        .metric .value {{ font-size: 24px; font-weight: bold; color: #58a6ff; }}
        .metric .label {{ font-size: 12px; color: #8b949e; margin-top: 5px; }}
        .positive {{ color: #3fb950 !important; }}
        .negative {{ color: #f85149 !important; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #30363d; }}
        th {{ background: #21262d; font-weight: 600; color: #58a6ff; }}
        .buy {{ background: rgba(63, 185, 80, 0.1); }}
        .sell {{ background: rgba(248, 81, 73, 0.1); }}
        .profit {{ color: #3fb950; font-weight: bold; }}
        .loss {{ color: #f85149; font-weight: bold; }}
        .chart-container {{ height: 300px; margin-top: 15px; }}
        .position-item {{ display: flex; justify-content: space-between; padding: 8px; background: #21262d; border-radius: 6px; margin-bottom: 8px; }}
        .position-item .ticker {{ color: #58a6ff; font-weight: bold; }}
        .position-item .shares {{ color: #8b949e; }}
        .footer {{ text-align: center; color: #8b949e; margin-top: 30px; font-size: 12px; }}
        .btn {{ display: inline-block; padding: 10px 20px; background: #238636; color: white; border-radius: 6px; text-decoration: none; margin-top: 15px; }}
        .btn:hover {{ background: #2ea043; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 投資組合回測報告</h1>
            <div class="subtitle">策略: {strategy_name} | 標的: {ticker_str} | 產生時間: {now}</div>
        </div>
        
        <div class="card">
            <h2>績效摘要</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="value {return_class}">{total_return:.2%}</div>
                    <div class="label">總報酬率</div>
                </div>
                <div class="metric">
                    <div class="value">{metrics.get('sharpe_ratio', 0):.2f}</div>
                    <div class="label">夏普比率</div>
                </div>
                <div class="metric">
                    <div class="value {dd_class}">{max_dd:.2%}</div>
                    <div class="label">最大回撤</div>
                </div>
                <div class="metric">
                    <div class="value">{metrics.get('win_rate', 0):.2%}</div>
                    <div class="label">勝率</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>資金變化</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="value">${metrics.get('initial_capital', 0):,.0f}</div>
                    <div class="label">初始資金</div>
                </div>
                <div class="metric">
                    <div class="value {return_class}">${metrics.get('final_equity', 0):,.0f}</div>
                    <div class="label">最終資金</div>
                </div>
                <div class="metric">
                    <div class="value">{metrics.get('total_trades', 0)}</div>
                    <div class="label">交易次數</div>
                </div>
                <div class="metric">
                    <div class="value">{profit_factor_str}</div>
                    <div class="label">盈虧比</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📈 權益曲線</h2>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>🏦 最終持倉</h2>
            {positions_html}
        </div>
        
        <div class="card">
            <h2>交易明細 (共 {len(trades_df)} 筆，最新在前)</h2>
            <table>
                <thead>
                    <tr>
                        <th>日期</th>
                        <th>股票</th>
                        <th>類型</th>
                        <th>價格</th>
                        <th>股數</th>
                        <th>損益</th>
                        <th>原因</th>
                    </tr>
                </thead>
                <tbody>
                    {trades_html}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            由 tw-quant-simple 投資組合引擎產生
        </div>
    </div>
    
    <script>
        const ctx = document.getElementById('equityChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {chart_dates},
                datasets: [{{
                    label: '權益',
                    data: {chart_values},
                    borderColor: '#58a6ff',
                    backgroundColor: 'rgba(88, 166, 255, 0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{
                        display: true,
                        ticks: {{ color: '#8b949e', maxTicksLimit: 10 }},
                        grid: {{ color: '#30363d' }}
                    }},
                    y: {{
                        display: true,
                        ticks: {{ color: '#8b949e' }},
                        grid: {{ color: '#30363d' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
'''
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✅ 投組報表已儲存: {save_path}")
        
    return html


def export_trades_csv(result: dict, save_path: str) -> str:
    """
    匯出交易明細為 CSV
    
    Args:
        result: PortfolioEngine.run() 的回傳結果
        save_path: 儲存路徑
        
    Returns:
        str: 儲存路徑
    """
    trades_df = result.get('trades', pd.DataFrame())
    
    if trades_df.empty:
        print("⚠️ 無交易記錄可匯出")
        return None
        
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    trades_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 交易明細已匯出: {save_path}")
    return save_path


def export_equity_curve_csv(result: dict, save_path: str) -> str:
    """
    匯出權益曲線為 CSV
    """
    equity_curve = result.get('equity_curve', pd.Series())
    
    if equity_curve.empty:
        print("⚠️ 無權益曲線可匯出")
        return None
        
    df = equity_curve.reset_index()
    df.columns = ['date', 'equity']
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 權益曲線已匯出: {save_path}")
    return save_path


if __name__ == '__main__':
    # 測試用
    print("📊 投組報表模組載入成功")
    print("用法：")
    print("  from backtest.portfolio_report import generate_portfolio_html_report")
    print("  generate_portfolio_html_report(result, tickers=['2330.TW'], save_path='reports/portfolio.html')")
