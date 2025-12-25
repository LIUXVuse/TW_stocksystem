# 技術規格文件 (SPEC.md)

本文件詳細說明系統的技術架構、模組功能和資料格式。

**最後更新**: 2024-12-25

---

## 系統架構

```
┌─────────────────────────────────────────────────────────────────┐
│                    Web UI (web/app.py + static/)                │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Backend  │  APScheduler  │  Static HTML/JS/CSS        │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│ /api/backtest│ /api/portfolio│ /api/optimize│ /api/download    │
├──────────────┴──────────────┴──────────────┴───────────────────┤
│                     回測引擎層 (backtest/)                       │
├────────┬────────┬────────┬────────┬────────┬──────────────────┤
│engine  │strategy│optimizer│portfolio│metrics│ strategy_config  │
│(核心)  │(12種)  │(優化)   │(投組)   │(指標) │ (策略說明)        │
├────────┴────────┴────────┴────────┴────────┴──────────────────┤
│                     資料處理層                                   │
├──────────────┬──────────────┬──────────────┬──────────────────┤
│data_loader   │ indicators   │institutional │ margin           │
│(資料整合)     │(技術指標)    │(法人資料)     │(融資融券)         │
├──────────────┴──────────────┴──────────────┴──────────────────┤
│                     資料儲存層 (data/)                          │
├──────────────┬──────────────┬──────────────────────────────────┤
│ tw-share/    │institutional/│ margin/                          │
│ (股價CSV)    │(法人JSON)    │(融資融券JSON)                     │
└──────────────┴──────────────┴──────────────────────────────────┘
```

---

## Web API 端點

### 資料下載

| 端點 | 方法 | 說明 |
|------|------|------|
| `/api/status` | GET | 取得系統狀態 |
| `/api/update/all` | POST | 一鍵更新全部資料 |
| `/api/download/stock` | POST | 下載股價資料 |
| `/api/download/institutional` | POST | 下載法人資料 |
| `/api/download/margin` | POST | 下載融資融券資料 |
| `/api/scan/market` | POST | 執行全市場掃描 |

### 回測 API

| 端點 | 方法 | 說明 |
|------|------|------|
| `/api/backtest/single` | POST | 單股回測（支援時間範圍選擇） |
| `/api/backtest/batch` | POST | 批次回測 |
| `/api/portfolio/run` | POST | 投資組合回測 |
| `/api/optimize` | POST | 參數優化 |

### 策略監控 API

| 端點 | 方法 | 說明 |
|------|------|------|
| `/api/monitor/signal` | POST | 取得特定股票+策略的當前訊號 |
| `/api/monitor/trades` | POST | 取得特定股票+策略的交易歷史 |
| `/api/monitor/list` | GET | 取得監控清單（伺服器端儲存） |
| `/api/monitor/list` | POST | 儲存監控清單（跨瀏覽器同步） |

### 回測請求格式

```json
// POST /api/backtest/single
{
  "ticker": "2330",
  "strategy": "MA5x20",
  "short_period": 5,
  "long_period": 20
}

// 回應
{
  "ticker": "2330.TW",
  "strategy": "MA5x20",
  "strategy_info": {
    "name": "MA5x20 均線交叉",
    "entry": "5日均線向上穿越20日均線（黃金交叉）",
    "exit": "5日均線向下穿越20日均線（死亡交叉）",
    "type": "趨勢追蹤",
    "risk": "中"
  },
  "metrics": {
    "total_return": 15.5,
    "sharpe_ratio": 1.23,
    "max_drawdown": -8.5,
    "win_rate": 55.0,
    "trade_count": 12
  },
  "trades": [
    {"date": "2024-01-15", "type": "BUY", "price": 580.0, "shares": 1000, "profit": null},
    {"date": "2024-02-20", "type": "SELL", "price": 610.0, "shares": 1000, "profit": 28500}
  ]
}
```

---

## 策略模組 (backtest/strategy.py)

### 策略基類

```python
class Strategy:
    def generate_signals(df) -> pd.Series:
        """回傳訊號序列: 1=買入, 0=持有, -1=賣出"""
```

### 內建策略 (12 種)

| 類別 | 參數 | 類型 |
|------|------|------|
| `MACrossStrategy` | short_period, long_period | 趨勢 |
| `RSIStrategy` | oversold, overbought | 動能 |
| `MACDStrategy` | 無 | 動能 |
| `BollingerStrategy` | 無 | 波動 |
| `MomentumBreakoutStrategy` | period, volume_mult | 突破 |
| `VolumeBreakoutStrategy` | volume_mult, price_change | 突破 |
| `TurtleStrategy` | entry_period, exit_period | 趨勢 |
| `MeanReversionStrategy` | ma_period, deviation | 均值回歸 |
| `InstitutionalFollowStrategy` | inst_type, consecutive_days | 籌碼 |
| `ChipTechStrategy` | inst_type | 籌碼 |

---

## 策略配置模組 (backtest/strategy_config.py)

### 策略說明字典

```python
STRATEGY_DESCRIPTIONS = {
    "MA5x20": {
        "name": "MA5x20 均線交叉",
        "entry": "5日均線向上穿越20日均線（黃金交叉）",
        "exit": "5日均線向下穿越20日均線（死亡交叉）",
        "type": "趨勢追蹤",
        "risk": "中"
    },
    # ... 其他策略
}
```

### 支援的策略名稱

- 技術分析：`MA5x20`, `MA5x60`, `MACD`, `RSI`, `動量突破`, `海龜策略`
- 法人籌碼：`外資連買3天`, `外資連買5天`, `投信連買3天`, `投信連買5天`
- 長期投資：`買入持有`, `定期定額`

---

## 投資組合策略 (backtest/strategy_portfolio.py)

### 策略類型

| 類別 | 說明 |
|------|------|
| `EqualWeightMonthlyStrategy` | 等權重再平衡（支援每週/每月/每季） |
| `BuyAndHoldStrategy` | 買入持有（支援三種模式） |
| `DCAStrategy` | 定期定額（每月指定日買入，無開盤自動順延） |
| `StrategyDrivenPortfolio` | 策略驅動（每檔股票用個別策略） |

### 等權重策略再平衡頻率

| 頻率 | 說明 |
|------|------|
| `weekly` | 每週一再平衡 |
| `monthly` | 每月第一個交易日再平衡（預設） |
| `quarterly` | 每季第一個交易日再平衡 |

### 買入持有策略三種模式

| 模式 | 說明 |
|------|------|
| `diamond` | 💎 鑽石手：永不賣出，忽略停損停利 |
| `rebuy` | 🔄 停損停利後買回：達標賣出 → 冷靜期後買回 |
| `multilayer` | 📈 多層次鑽石手：永不賣出 + 自訂加碼時間表 |

#### 多層次鑽石手加碼時間表格式

```json
{
  "extra_buys": [
    {"date": "2024-12-05", "amount": 1000000},
    {"date": "2025-04-03", "amount": 300000}
  ]
}
```

- 如果指定日期無開盤，自動順延到下一個交易日

---

## 資料格式

### 股價 CSV (data/tw-share/dayK/*.csv)

```csv
date,open,high,low,close,volume,dividends,stock splits,ma5,ma10,...
2024-01-02,580.0,585.0,578.0,582.0,25000000,0,0,580.5,579.2,...
```

### 法人 JSON (data/institutional/*.json)

```json
{
  "2330.TW": {
    "foreign": 5000000,
    "trust": -200000,
    "dealer": 100000,
    "total": 4900000
  }
}
```

- 單位：股（非張）
- 正數=買超，負數=賣超

---

## 開發路線圖

### ✅ Phase 1: 資料基礎

- [x] 股價下載（增量更新）
- [x] 技術指標計算
- [x] 法人資料下載
- [x] 融資融券資料

### ✅ Phase 2: 回測引擎

- [x] 回測核心
- [x] 12 種內建策略（含法人籌碼）
- [x] 績效指標
- [x] 參數優化
- [x] 批次回測

### ✅ Phase 3: 訊號系統

- [x] 訊號掃描器
- [x] 動態策略排名
- [x] HTML 報表
- [x] 夏普比率全市場掃描

### ✅ Phase 4: Web UI

- [x] FastAPI 後端
- [x] 單股回測 API
- [x] 投資組合回測 API
- [x] 參數優化 API
- [x] 策略說明顯示
- [x] 交易明細顯示
- [x] 一鍵更新功能
- [x] 自動排程（每日 19:00）

### ✅ Phase 5: 進階功能

- [x] 回測時間範圍選擇（開始/結束日期）
- [x] 策略監控頁面（12 種策略）
- [x] 閃退問題修復（全市場掃描/投資組合）
- [ ] 止損/止利/移動停損設定 UI
- [ ] 權益曲線圖表視覺化
- [ ] Line/Telegram 通知

### 🔜 Phase 6: 未來規劃

- [ ] 投資組合每檔股票獨立策略設定
- [ ] 權益曲線圖表視覺化
- [ ] Line/Telegram 通知

---

## 效能參考

| 操作 | 時間 |
|------|------|
| 全市場下載（增量） | 15-20 分鐘 |
| 法人歷史下載（1年） | 10-15 分鐘 |
| 技術指標計算 | 5-10 分鐘 |
| 訊號掃描 | 3-5 分鐘 |
| 夏普比率掃描 | 30-60 分鐘 |
| 單股回測 | < 1 秒 |
| 參數優化 (16組合) | 2-3 秒 |

---

## 依賴套件

```
pandas>=1.5.0
numpy>=1.24.0
yfinance>=0.2.28
requests>=2.28.0
tqdm>=4.64.0
lxml>=4.9.0
fastapi>=0.100.0
uvicorn>=0.23.0
apscheduler>=3.10.0
pydantic>=2.0.0
```
