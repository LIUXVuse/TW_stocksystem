# -*- coding: utf-8 -*-
"""
台股量化系統 Web UI
FastAPI 後端 + 靜態前端
"""
import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

# 關閉 uvicorn access log（那些 GET /api/status 200 OK）
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# 專案根目錄
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
sys.path.insert(0, BASE_DIR)

# 全域狀態
app_state = {
    "scheduler_enabled": False,
    "last_update": None,
    "running_task": None,
    "task_progress": 0,
    "current_step": "",
    "logs": []
}

scheduler = AsyncIOScheduler()

def add_log(message: str, level: str = "info"):
    """新增執行日誌"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"time": timestamp, "message": message, "level": level}
    app_state["logs"].insert(0, log_entry)
    app_state["logs"] = app_state["logs"][:50]
    print(f"[{timestamp}] {message}")

async def run_script_async(script_name: str, args: list = None) -> tuple:
    """非同步執行 Python 腳本，捕捉即時輸出"""
    cmd = [sys.executable, os.path.join(BASE_DIR, script_name)]
    if args:
        cmd.extend(args)
    
    add_log(f"🔄 開始執行: {script_name}")
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=BASE_DIR
        )
        
        # 即時讀取輸出並加入日誌
        output_lines = []
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            decoded = line.decode().strip()
            if decoded:
                output_lines.append(decoded)
                # 只記錄有進度的行
                if any(x in decoded for x in ['%', '成功', '失敗', '完成', '更新', '下載']):
                    app_state["current_step"] = decoded[:100]
                    add_log(decoded[:100])
        
        await process.wait()
        
        if process.returncode == 0:
            add_log(f"✅ {script_name} 執行成功", "success")
            return True, "\n".join(output_lines)
        else:
            stderr = await process.stderr.read()
            add_log(f"❌ {script_name} 執行失敗", "error")
            return False, stderr.decode()
            
    except asyncio.TimeoutError:
        add_log(f"⏰ {script_name} 執行超時", "error")
        return False, "Timeout"
    except Exception as e:
        add_log(f"❌ {script_name} 執行錯誤: {e}", "error")
        return False, str(e)

async def daily_update_task():
    """每日自動更新任務"""
    add_log("🔄 開始每日自動更新")
    app_state["running_task"] = "daily_update"
    app_state["task_progress"] = 0
    
    try:
        # 1. 下載股價 (20%)
        add_log("📥 [1/5] 下載股價中...")
        await run_script_async("downloader_tw.py")
        app_state["task_progress"] = 20
        
        # 2. 下載法人 (40%)
        add_log("📥 [2/5] 下載法人中...")
        await run_script_async("institutional.py", ["auto"])
        app_state["task_progress"] = 40
        
        # 3. 下載融資融券 (60%)
        add_log("📥 [3/5] 下載融資融券中...")
        await run_script_async("margin.py", ["auto"])
        app_state["task_progress"] = 60
        
        # 4. 計算指標 (80%)
        add_log("📊 [4/5] 計算指標中...")
        await run_script_async("indicators.py")
        app_state["task_progress"] = 80
        
        # 5. 訊號掃描 (100%)
        add_log("🔍 [5/5] 訊號掃描中...")
        await run_script_async("signal_scanner.py")
        app_state["task_progress"] = 100
        
        app_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        add_log("✅ 每日自動更新完成", "success")
    except Exception as e:
        add_log(f"❌ 更新失敗: {e}", "error")
    finally:
        app_state["running_task"] = None
        app_state["current_step"] = ""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    scheduler.start()
    add_log("🚀 系統啟動")
    yield
    scheduler.shutdown()

app = FastAPI(title="台股量化系統", lifespan=lifespan)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 靜態檔案
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 報告目錄
reports_dir = os.path.join(BASE_DIR, "reports")
if os.path.exists(reports_dir):
    app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")

# ==================== API Routes ====================

@app.get("/")
async def index():
    """首頁"""
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/api/status")
async def get_status():
    """取得系統狀態"""
    return {
        "scheduler_enabled": app_state["scheduler_enabled"],
        "last_update": app_state["last_update"],
        "running_task": app_state["running_task"],
        "task_progress": app_state["task_progress"],
        "current_step": app_state["current_step"],
        "logs": app_state["logs"][:20]
    }

@app.post("/api/scheduler/toggle")
async def toggle_scheduler():
    """開關排程"""
    if app_state["scheduler_enabled"]:
        try:
            scheduler.remove_job("daily_update")
        except:
            pass
        app_state["scheduler_enabled"] = False
        add_log("⏸️ 自動排程已關閉")
    else:
        scheduler.add_job(
            daily_update_task,
            'cron',
            hour=19,
            minute=0,
            id="daily_update",
            replace_existing=True
        )
        app_state["scheduler_enabled"] = True
        add_log("▶️ 自動排程已開啟 (每日 19:00)")
    
    return {"scheduler_enabled": app_state["scheduler_enabled"]}

# ==================== 下載 API ====================

@app.post("/api/download/all")
async def download_all():
    """一鍵下載所有資料"""
    if app_state["running_task"]:
        raise HTTPException(400, "已有任務執行中")
    
    # 使用 asyncio.create_task 正確啟動背景任務
    asyncio.create_task(daily_update_task())
    return {"message": "開始一鍵下載"}

@app.post("/api/download/price")
async def download_price():
    """手動下載股價"""
    if app_state["running_task"]:
        raise HTTPException(400, "已有任務執行中")
    
    async def task():
        app_state["running_task"] = "download_price"
        add_log("📥 開始下載股價...")
        try:
            await run_script_async("downloader_tw.py")
            app_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        finally:
            app_state["running_task"] = None
            app_state["current_step"] = ""
    
    asyncio.create_task(task())
    return {"message": "開始下載股價"}

@app.post("/api/download/institutional")
async def download_institutional():
    """手動下載法人"""
    if app_state["running_task"]:
        raise HTTPException(400, "已有任務執行中")
    
    async def task():
        app_state["running_task"] = "download_institutional"
        add_log("📥 開始下載法人資料...")
        try:
            await run_script_async("institutional.py", ["auto"])
        finally:
            app_state["running_task"] = None
            app_state["current_step"] = ""
    
    asyncio.create_task(task())
    return {"message": "開始下載法人"}

@app.post("/api/download/margin")
async def download_margin():
    """手動下載融資融券"""
    if app_state["running_task"]:
        raise HTTPException(400, "已有任務執行中")
    
    async def task():
        app_state["running_task"] = "download_margin"
        add_log("📥 開始下載融資融券...")
        try:
            await run_script_async("margin.py", ["auto"])
        finally:
            app_state["running_task"] = None
            app_state["current_step"] = ""
    
    asyncio.create_task(task())
    return {"message": "開始下載融資融券"}

@app.post("/api/download/indicators")
async def calculate_indicators():
    """計算指標"""
    if app_state["running_task"]:
        raise HTTPException(400, "已有任務執行中")
    
    async def task():
        app_state["running_task"] = "calculate_indicators"
        add_log("📊 開始計算指標...")
        try:
            await run_script_async("indicators.py")
        finally:
            app_state["running_task"] = None
            app_state["current_step"] = ""
    
    asyncio.create_task(task())
    return {"message": "開始計算指標"}

@app.post("/api/scan/signals")
async def scan_signals():
    """執行訊號掃描"""
    if app_state["running_task"]:
        raise HTTPException(400, "已有任務執行中")
    
    async def task():
        app_state["running_task"] = "scan_signals"
        add_log("🔍 開始訊號掃描...")
        try:
            await run_script_async("signal_scanner.py")
        finally:
            app_state["running_task"] = None
            app_state["current_step"] = ""
    
    asyncio.create_task(task())
    return {"message": "開始訊號掃描"}

@app.post("/api/scan/market")
async def scan_market():
    """執行全市場夏普比率掃描（需要較長時間）"""
    if app_state["running_task"]:
        raise HTTPException(400, "已有任務執行中")
    
    async def task():
        app_state["running_task"] = "scan_market"
        add_log("🔍 開始全市場掃描（預計 30-60 分鐘）...")
        try:
            await run_script_async("scan_market.py")
            add_log("✅ 全市場掃描完成", "success")
        finally:
            app_state["running_task"] = None
            app_state["current_step"] = ""
    
    asyncio.create_task(task())
    return {"message": "開始全市場掃描"}

# ==================== 訊號報告 ====================

@app.get("/api/signals/today")
async def get_today_signals():
    """取得今日訊號"""
    report_path = os.path.join(BASE_DIR, "reports", "signal_alert.html")
    if os.path.exists(report_path):
        return FileResponse(report_path)
    raise HTTPException(404, "報告尚未產生，請先執行訊號掃描")

# ==================== 股票清單 ====================

@app.get("/api/tickers")
async def get_tickers():
    """取得所有股票代碼"""
    data_dir = os.path.join(BASE_DIR, "data", "tw-share", "dayK")
    tickers = []
    
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith(".csv"):
                parts = f.replace(".csv", "").split("_")
                ticker = parts[0]
                name = parts[1] if len(parts) > 1 else ticker
                tickers.append({"ticker": ticker, "name": name})
    
    tickers.sort(key=lambda x: x["ticker"])
    return {"tickers": tickers}

# ==================== 回測 API ====================

class BacktestRequest(BaseModel):
    ticker: str
    strategy: str = "MA5x20"
    capital: int = 500000  # 初始資金（預設 50 萬）
    short_period: Optional[int] = 5
    long_period: Optional[int] = 20
    start_date: Optional[str] = None  # 格式: YYYY-MM-DD
    end_date: Optional[str] = None    # 格式: YYYY-MM-DD

def normalize_ticker(ticker: str) -> str:
    """標準化股票代碼，自動補上 .TW 或 .TWO"""
    ticker = ticker.strip().upper()
    
    # 如果已經有後綴，直接返回
    if '.TW' in ticker:
        return ticker
    
    # 純數字，嘗試加 .TW
    return f"{ticker}.TW"

@app.post("/api/backtest/single")
async def run_single_backtest(req: BacktestRequest):
    """執行單股回測"""
    try:
        from glob import glob
        from data_loader import load_stock_with_institutional, find_stock_file
        from backtest.engine import BacktestEngine
        from backtest.strategy import (
            MACrossStrategy, RSIStrategy, MACDStrategy,
            MomentumBreakoutStrategy, TurtleStrategy,
            InstitutionalFollowStrategy, BollingerStrategy,
            VolumeBreakoutStrategy
        )
        
        # 標準化股票代碼
        ticker = normalize_ticker(req.ticker)
        
        # 先嘗試 .TW，不行再試 .TWO
        csv_path = find_stock_file(ticker)
        if not csv_path and ticker.endswith('.TW'):
            ticker = ticker.replace('.TW', '.TWO')
            csv_path = find_stock_file(ticker)
        
        if not csv_path:
            raise HTTPException(404, f"找不到 {req.ticker} 的資料")
        
        import pandas as pd  # 確保 pd 在所有情況下都可用
        
        # 判斷是否需要法人資料
        needs_institutional = req.strategy in ['外資連買', '投信連買', '外資連買3天', '外資連買5天', '投信連買3天', '投信連買5天']
        
        if needs_institutional:
            # 使用 data_loader 載入包含法人資料的股票
            df = load_stock_with_institutional(ticker)
        else:
            df = pd.read_csv(csv_path)
            df.columns = [c.lower() for c in df.columns]
        
        df = df.sort_values('date')
        
        # 根據日期範圍過濾數據
        if req.start_date:
            df = df[df['date'] >= req.start_date]
        if req.end_date:
            df = df[df['date'] <= req.end_date]
        
        if len(df) < 30:
            raise HTTPException(400, f"數據不足，範圍內只有 {len(df)} 筆數據（最少需要 30 筆）")
        
        strategy_map = {
            "MA5x20": MACrossStrategy(5, 20),
            "MA5x60": MACrossStrategy(5, 60),
            "RSI": RSIStrategy(),
            "MACD": MACDStrategy(),
            "布林通道": BollingerStrategy(),
            "動量突破": MomentumBreakoutStrategy(),
            "量價突破": VolumeBreakoutStrategy(),
            "海龜策略": TurtleStrategy(),
            # 法人策略（不同連買天數）
            "外資連買3天": InstitutionalFollowStrategy('foreign', consecutive_days=3),
            "外資連買5天": InstitutionalFollowStrategy('foreign', consecutive_days=5),
            "投信連買3天": InstitutionalFollowStrategy('trust', consecutive_days=3),
            "投信連買5天": InstitutionalFollowStrategy('trust', consecutive_days=5),
            # 舊的預設（5天）
            "外資連買": InstitutionalFollowStrategy('foreign', consecutive_days=5),
            "投信連買": InstitutionalFollowStrategy('trust', consecutive_days=5),
            "MA_Custom": MACrossStrategy(req.short_period or 5, req.long_period or 20),
        }
        strategy = strategy_map.get(req.strategy, MACrossStrategy(5, 20))
        
        # 自訂均線時，動態計算 MA 欄位
        if req.strategy == "MA_Custom":
            short_p = req.short_period or 5
            long_p = req.long_period or 20
            price_col = 'close' if 'close' in df.columns else 'Close'
            df[f'ma{short_p}'] = df[price_col].rolling(short_p).mean()
            df[f'ma{long_p}'] = df[price_col].rolling(long_p).mean()
        
        # 使用用戶指定的初始資金
        engine = BacktestEngine(initial_capital=req.capital)
        result = engine.run(df, strategy)
        metrics = result.get('metrics', {})
        trades_raw = result.get('trades', [])
        
        # 取得策略說明
        from backtest.strategy_config import get_strategy_description
        strategy_desc = get_strategy_description(req.strategy)
        
        # 準備交易明細（最近 30 筆）
        trades_list = []
        
        # 處理 trades（可能是 DataFrame 或 list）
        if isinstance(trades_raw, pd.DataFrame) and not trades_raw.empty:
            for _, row in trades_raw.tail(30).iterrows():
                trades_list.append({
                    'date': str(row.get('date', ''))[:10],
                    'type': row.get('type', ''),
                    'price': round(row.get('price', 0), 2),
                    'shares': int(row.get('shares', 0)),
                    'profit': round(row.get('profit', 0), 0) if row.get('type') == 'SELL' else None
                })
        elif isinstance(trades_raw, list) and len(trades_raw) > 0:
            for t in trades_raw[-30:]:
                trades_list.append({
                    'date': str(t.get('date', ''))[:10],
                    'type': t.get('type', ''),
                    'price': round(t.get('price', 0), 2),
                    'shares': int(t.get('shares', 0)),
                    'profit': round(t.get('profit', 0), 0) if t.get('type') == 'SELL' else None
                })
        
        return {
            "ticker": ticker,
            "strategy": req.strategy,
            "strategy_info": {
                "name": strategy_desc.get('name', req.strategy),
                "entry": strategy_desc.get('entry', ''),
                "exit": strategy_desc.get('exit', ''),
                "type": strategy_desc.get('type', ''),
                "risk": strategy_desc.get('risk', '')
            },
            "metrics": {
                "total_return": round(metrics.get('total_return', 0) * 100, 2),
                "sharpe_ratio": round(metrics.get('sharpe_ratio', 0), 2),
                "max_drawdown": round(metrics.get('max_drawdown', 0) * 100, 2),
                "win_rate": round(metrics.get('win_rate', 0) * 100, 2),
                "trade_count": metrics.get('trade_count', 0)
            },
            "trades": trades_list
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

# ==================== 批次回測 ====================

class BatchBacktestRequest(BaseModel):
    tickers: List[str]
    strategy: str = "MA5x20"
    capital: int = 500000  # 初始資金（預設 50 萬）

@app.post("/api/backtest/batch")
async def run_batch_backtest(req: BatchBacktestRequest):
    """執行多股回測"""
    results = []
    for ticker in req.tickers[:20]:
        try:
            single_req = BacktestRequest(ticker=ticker, strategy=req.strategy, capital=req.capital)
            result = await run_single_backtest(single_req)
            results.append(result)
        except:
            continue
    
    results.sort(key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
    return {"results": results}

# ==================== 投資組合 ====================

class PortfolioRequest(BaseModel):
    tickers: List[str]
    initial_capital: float = 1000000
    strategy: str = "equal_weight"  # equal_weight, buy_hold, dca
    stop_loss: Optional[float] = None  # 停損 (如 -0.10)
    take_profit: Optional[float] = None  # 停利 (如 0.30)
    dca_day: int = 1  # 定期定額買入日 (1-28)
    dca_amount: Optional[float] = None  # 定期定額每月投入金額
    start_date: Optional[str] = None  # 開始日期 YYYY-MM-DD
    end_date: Optional[str] = None  # 結束日期 YYYY-MM-DD
    # 等權重策略參數
    rebalance_freq: str = "monthly"  # weekly, monthly, quarterly
    # 買入持有策略參數
    buy_hold_mode: str = "diamond"  # diamond, rebuy, multilayer
    cooldown_days: int = 30  # 冷靜期天數
    rebuy_amount: str = "all"  # all, original
    extra_buys: Optional[List[dict]] = None  # 多層次鑽石手加碼時間表 [{"date": "2024-12-05", "amount": 100000}]

@app.post("/api/portfolio/run")
async def run_portfolio_backtest(req: PortfolioRequest):
    """執行投組回測"""
    try:
        import pandas as pd
        from glob import glob
        from datetime import datetime
        from backtest.portfolio import PortfolioEngine
        from backtest.strategy_portfolio import (
            EqualWeightMonthlyStrategy,
            BuyAndHoldStrategy,
            DCAStrategy
        )
        from backtest.portfolio_report import generate_portfolio_html_report
        
        data_map = {}
        for ticker in req.tickers[:10]:
            data_path = os.path.join(BASE_DIR, "data", "tw-share", "dayK", f"{ticker}*.csv")
            files = glob(data_path)
            if files:
                df = pd.read_csv(files[0])
                date_col = 'date' if 'date' in df.columns else 'Date'
                df = df.sort_values(date_col)
                df[date_col] = df[date_col].astype(str)
                
                # 依日期範圍過濾
                if req.start_date:
                    df = df[df[date_col] >= req.start_date]
                if req.end_date:
                    df = df[df[date_col] <= req.end_date]
                
                if len(df) > 30:  # 確保有足夠資料
                    data_map[ticker] = df
        
        if not data_map:
            raise HTTPException(404, "找不到任何股票資料（或日期範圍內資料不足）")
        
        # 根據策略選擇建立策略實例
        strategy_name = ""
        if req.strategy == "buy_hold":
            mode = req.buy_hold_mode or "diamond"
            
            if mode == "diamond":
                # 鑽石手：永不賣出，不需要停損停利
                strategy = BuyAndHoldStrategy(
                    stop_loss=None,
                    take_profit=None,
                    mode="diamond"
                )
                strategy_name = "💎 鑽石手（永不賣出）"
            elif mode == "rebuy":
                # 停損停利後買回
                strategy = BuyAndHoldStrategy(
                    stop_loss=req.stop_loss,
                    take_profit=req.take_profit,
                    mode="rebuy",
                    cooldown_days=req.cooldown_days,
                    rebuy_amount=req.rebuy_amount
                )
                strategy_name = f"🔄 停損停利後買回 (冷靜{req.cooldown_days}天)"
            elif mode == "multilayer":
                # 多層次鑽石手
                strategy = BuyAndHoldStrategy(
                    stop_loss=None,
                    take_profit=None,
                    mode="multilayer",
                    extra_buys=req.extra_buys
                )
                extra_count = len(req.extra_buys) if req.extra_buys else 0
                strategy_name = f"📈 多層次鑽石手 ({extra_count} 筆加碼)"
            else:
                # 預設鑽石手
                strategy = BuyAndHoldStrategy(mode="diamond")
                strategy_name = "💎 鑽石手"
                
        elif req.strategy == "dca":
            monthly_amt = req.dca_amount or 10000  # 預設每月 1 萬元
            strategy = DCAStrategy(buy_day=req.dca_day, monthly_amount=monthly_amt)
            strategy_name = f"📅 定期定額 (每月{req.dca_day}日, ${monthly_amt:,.0f})"
        else:
            # 等權重策略
            freq = req.rebalance_freq or "monthly"
            freq_label = {"weekly": "每週", "monthly": "每月", "quarterly": "每季"}.get(freq, "每月")
            strategy = EqualWeightMonthlyStrategy(freq=freq)
            strategy_name = f"📊 等權重 ({freq_label}再平衡)"
        
        engine = PortfolioEngine(initial_capital=req.initial_capital)
        result = engine.run(data_map, strategy)
        metrics = result.get('metrics', {})
        
        # 使用正式報告產生器 (帶時間戳檔名)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"portfolio_{timestamp}.html"
        report_path = os.path.join(REPORTS_DIR, report_filename)
        
        # 產生詳細 HTML 報告 (包含圖表和交易明細)
        generate_portfolio_html_report(
            result, 
            tickers=list(data_map.keys()), 
            strategy_name=strategy_name, 
            save_path=report_path
        )
        
        # 取得權益曲線的最後值
        equity_curve = result.get('equity_curve', None)
        if equity_curve is not None and len(equity_curve) > 0:
            final_value = float(equity_curve.iloc[-1]) if hasattr(equity_curve, 'iloc') else float(equity_curve[-1])
        else:
            final_value = req.initial_capital
        
        return {
            "tickers": list(data_map.keys()),
            "strategy": strategy_name,
            "metrics": {
                "total_return": round(metrics.get('total_return', 0) * 100, 2),
                "sharpe_ratio": round(metrics.get('sharpe_ratio', 0), 2),
                "max_drawdown": round(metrics.get('max_drawdown', 0) * 100, 2),
                "win_rate": round(metrics.get('win_rate', 0) * 100, 2),
                "trade_count": metrics.get('trade_count', 0)
            },
            "final_value": round(final_value, 0),
            "report_url": f"/reports/{report_filename}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

# ==================== 報告管理 ====================

@app.get("/api/reports")
async def list_reports(report_type: str = None):
    """列出所有報告"""
    try:
        reports = []
        if os.path.exists(REPORTS_DIR):
            for filename in os.listdir(REPORTS_DIR):
                if not filename.endswith('.html'):
                    continue
                    
                # 過濾報告類型
                if report_type:
                    if report_type == "portfolio" and not filename.startswith("portfolio_"):
                        continue
                    elif report_type == "scan" and not filename.startswith("market_scan"):
                        continue
                    elif report_type == "signal" and not filename.startswith("signal_"):
                        continue
                
                filepath = os.path.join(REPORTS_DIR, filename)
                stat = os.stat(filepath)
                
                # 判斷報告類型
                if filename.startswith("portfolio_"):
                    rtype = "投組回測"
                    icon = "💼"
                elif filename.startswith("market_scan"):
                    rtype = "全市場掃描"
                    icon = "🔍"
                elif filename.startswith("signal_"):
                    rtype = "今日訊號"
                    icon = "📊"
                else:
                    rtype = "其他"
                    icon = "📄"
                
                reports.append({
                    "filename": filename,
                    "type": rtype,
                    "icon": icon,
                    "url": f"/reports/{filename}",
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
        
        # 按修改時間排序（最新的在前）
        reports.sort(key=lambda x: x['modified'], reverse=True)
        return {"reports": reports}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/api/reports/{filename}")
async def delete_report(filename: str):
    """刪除指定報告"""
    try:
        # 安全性檢查：只能刪除 .html 和 .csv 檔案
        if not (filename.endswith('.html') or filename.endswith('.csv')):
            raise HTTPException(400, "只能刪除報告檔案")
        
        # 防止路徑遍歷攻擊
        if '..' in filename or '/' in filename:
            raise HTTPException(400, "無效的檔案名稱")
        
        filepath = os.path.join(REPORTS_DIR, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return {"success": True, "message": f"已刪除 {filename}"}
        else:
            raise HTTPException(404, "找不到該報告")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

# ==================== 參數優化 ====================

class OptimizeRequest(BaseModel):
    ticker: str
    short_range: List[int] = [5, 10, 15, 20]
    long_range: List[int] = [20, 40, 60, 120]

@app.post("/api/optimize")
async def run_optimization(req: OptimizeRequest):
    """執行參數優化"""
    try:
        import pandas as pd
        import numpy as np
        from glob import glob
        from backtest.optimizer import StrategyOptimizer
        from backtest.strategy import MACrossStrategy
        
        data_path = os.path.join(BASE_DIR, "data", "tw-share", "dayK", f"{req.ticker}*.csv")
        files = glob(data_path)
        if not files:
            raise HTTPException(404, f"找不到 {req.ticker} 的資料")
        
        df = pd.read_csv(files[0])
        df = df.sort_values('date' if 'date' in df.columns else 'Date')
        
        optimizer = StrategyOptimizer(min_trades=3)
        results = optimizer.grid_search(
            df,
            MACrossStrategy,
            param_grid={
                'short_period': req.short_range,
                'long_period': req.long_range
            },
            metric='sharpe_ratio'
        )
        
        # 處理 inf/nan 值，替換為 None（JSON 可序列化）
        results = results.replace([np.inf, -np.inf], np.nan)
        top_results = results.head(10).to_dict('records')
        
        # 將 nan 轉為 None
        for record in top_results:
            for key, value in record.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    record[key] = None
        
        return {
            "ticker": req.ticker,
            "best_params": top_results[0] if top_results else {},
            "all_results": top_results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ==================== 策略監控 ====================

class MonitorRequest(BaseModel):
    ticker: str
    strategy: str = "MA5x20"
    short_period: Optional[int] = None
    long_period: Optional[int] = None

@app.post("/api/monitor/signal")
async def get_monitor_signal(req: MonitorRequest):
    """取得特定股票+策略的當前訊號"""
    try:
        import pandas as pd
        from glob import glob
        from backtest.strategy import (
            MACrossStrategy, RSIStrategy, MACDStrategy, 
            BollingerStrategy, MomentumBreakoutStrategy, VolumeBreakoutStrategy, TurtleStrategy,
            InstitutionalFollowStrategy
        )
        
        ticker = normalize_ticker(req.ticker)
        data_path = os.path.join(BASE_DIR, "data", "tw-share", "dayK", f"{ticker.replace('.TW', '')}*.csv")
        files = glob(data_path)
        
        if not files:
            raise HTTPException(404, f"找不到 {ticker} 股票資料")
        
        df = pd.read_csv(files[0])
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_values('date').tail(100)  # 取最近 100 筆計算
        
        # 計算所需指標
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        
        strategy_map = {
            "MA5x20": MACrossStrategy(5, 20),
            "MA5x60": MACrossStrategy(5, 60),
            "RSI": RSIStrategy(),
            "MACD": MACDStrategy(),
            "布林通道": BollingerStrategy(),
            "動量突破": MomentumBreakoutStrategy(),
            "量價突破": VolumeBreakoutStrategy(),
            "海龜策略": TurtleStrategy(),
            "外資連貰3天": InstitutionalFollowStrategy('foreign', consecutive_days=3),
            "外資連貰5天": InstitutionalFollowStrategy('foreign', consecutive_days=5),
            "投信連貰3天": InstitutionalFollowStrategy('trust', consecutive_days=3),
            "投信連貰5天": InstitutionalFollowStrategy('trust', consecutive_days=5),
        }
        
        # 支援自訂均線
        if req.short_period and req.long_period:
            strategy = MACrossStrategy(req.short_period, req.long_period)
            # 計算自訂均線
            df[f'ma{req.short_period}'] = df['close'].rolling(req.short_period).mean()
            df[f'ma{req.long_period}'] = df['close'].rolling(req.long_period).mean()
        elif req.strategy.startswith('MA') and 'x' in req.strategy:
            # 解析 MA10x40 格式
            parts = req.strategy.replace('MA', '').split('x')
            if len(parts) == 2:
                short_p = int(parts[0])
                long_p = int(parts[1])
                strategy = MACrossStrategy(short_p, long_p)
                df[f'ma{short_p}'] = df['close'].rolling(short_p).mean()
                df[f'ma{long_p}'] = df['close'].rolling(long_p).mean()
            else:
                strategy = strategy_map.get(req.strategy, MACrossStrategy(5, 20))
        else:
            strategy = strategy_map.get(req.strategy, MACrossStrategy(5, 20))
        
        signals = strategy.generate_signals(df)
        
        # 取得最後一個訊號
        last_signal = signals.iloc[-1] if len(signals) > 0 else 0
        last_date = df['date'].iloc[-1] if len(df) > 0 else None
        
        return {
            "signal": int(last_signal),
            "last_date": str(last_date) if last_date else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/monitor/trades")
async def get_monitor_trades(req: MonitorRequest):
    """取得特定股票+策略的交易歷史"""
    try:
        import pandas as pd
        from glob import glob
        from backtest.strategy import (
            MACrossStrategy, RSIStrategy, MACDStrategy, 
            BollingerStrategy, MomentumBreakoutStrategy, VolumeBreakoutStrategy, TurtleStrategy,
            InstitutionalFollowStrategy
        )
        
        ticker = normalize_ticker(req.ticker)
        data_path = os.path.join(BASE_DIR, "data", "tw-share", "dayK", f"{ticker.replace('.TW', '')}*.csv")
        files = glob(data_path)
        
        if not files:
            raise HTTPException(404, f"找不到 {ticker} 股票資料")
        
        df = pd.read_csv(files[0])
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_values('date').reset_index(drop=True)
        
        # 計算所需指標
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        
        strategy_map = {
            "MA5x20": MACrossStrategy(5, 20),
            "MA5x60": MACrossStrategy(5, 60),
            "RSI": RSIStrategy(),
            "MACD": MACDStrategy(),
            "布林通道": BollingerStrategy(),
            "動量突破": MomentumBreakoutStrategy(),
            "量價突破": VolumeBreakoutStrategy(),
            "海龜策略": TurtleStrategy(),
            "外資連買3天": InstitutionalFollowStrategy('foreign', consecutive_days=3),
            "外資連買5天": InstitutionalFollowStrategy('foreign', consecutive_days=5),
            "投信連買3天": InstitutionalFollowStrategy('trust', consecutive_days=3),
            "投信連買5天": InstitutionalFollowStrategy('trust', consecutive_days=5),
        }
        
        # 支援自訂均線
        if req.short_period and req.long_period:
            strategy = MACrossStrategy(req.short_period, req.long_period)
            df[f'ma{req.short_period}'] = df['close'].rolling(req.short_period).mean()
            df[f'ma{req.long_period}'] = df['close'].rolling(req.long_period).mean()
        elif req.strategy.startswith('MA') and 'x' in req.strategy:
            parts = req.strategy.replace('MA', '').split('x')
            if len(parts) == 2:
                short_p = int(parts[0])
                long_p = int(parts[1])
                strategy = MACrossStrategy(short_p, long_p)
                df[f'ma{short_p}'] = df['close'].rolling(short_p).mean()
                df[f'ma{long_p}'] = df['close'].rolling(long_p).mean()
            else:
                strategy = strategy_map.get(req.strategy, MACrossStrategy(5, 20))
        else:
            strategy = strategy_map.get(req.strategy, MACrossStrategy(5, 20))
        signals = strategy.generate_signals(df)
        
        # 解析交易
        trades = []
        buy_date = None
        buy_price = None
        
        for i in range(len(signals)):
            if signals.iloc[i] == 1:  # 買入
                buy_date = df['date'].iloc[i]
                buy_price = df['close'].iloc[i]
                trades.append({
                    "date": str(buy_date),
                    "type": "buy",
                    "price": float(buy_price),
                    "holding_days": None,
                    "return_pct": None
                })
            elif signals.iloc[i] == -1 and buy_date is not None:  # 賣出
                sell_date = df['date'].iloc[i]
                sell_price = df['close'].iloc[i]
                holding_days = i - df[df['date'] == buy_date].index[0]
                return_pct = (sell_price - buy_price) / buy_price if buy_price else 0
                
                trades.append({
                    "date": str(sell_date),
                    "type": "sell",
                    "price": float(sell_price),
                    "holding_days": int(holding_days),
                    "return_pct": float(return_pct)
                })
                buy_date = None
                buy_price = None
        
        return {"trades": trades[-20:]}  # 返回最近 20 筆交易
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ==================== 監控清單持久化 ====================

MONITOR_LIST_FILE = os.path.join(BASE_DIR, "data", "monitor_list.json")

@app.get("/api/monitor/list")
async def get_monitor_list():
    """取得監控清單"""
    try:
        if os.path.exists(MONITOR_LIST_FILE):
            import json
            with open(MONITOR_LIST_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {"list": data.get("list", [])}
        return {"list": []}
    except Exception as e:
        return {"list": [], "error": str(e)}

class MonitorListRequest(BaseModel):
    list: List[dict]  # [{"ticker": "2330", "strategy": "MA5x20"}, ...]

@app.post("/api/monitor/list")
async def save_monitor_list(req: MonitorListRequest):
    """儲存監控清單"""
    try:
        import json
        os.makedirs(os.path.dirname(MONITOR_LIST_FILE), exist_ok=True)
        with open(MONITOR_LIST_FILE, 'w', encoding='utf-8') as f:
            json.dump({"list": req.list}, f, ensure_ascii=False, indent=2)
        return {"success": True, "count": len(req.list)}
    except Exception as e:
        raise HTTPException(500, f"儲存失敗: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
