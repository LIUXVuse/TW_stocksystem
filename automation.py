# -*- coding: utf-8 -*-
"""
自動化流水線腳本
每日自動執行: 下載 → 掃描 → 通知
"""
import os
import sys
from datetime import datetime, timedelta
import subprocess

# 取得專案根目錄
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from notifier import notify

# =====================
# 交易日判斷
# =====================
def is_trading_day(date: datetime = None) -> bool:
    """
    判斷是否為交易日 (簡易版: 排除週末)
    實務上應整合台灣證交所公布的休市日
    """
    date = date or datetime.now()
    
    # 週末不交易
    if date.weekday() >= 5:  # 0=Mon, 5=Sat, 6=Sun
        return False
        
    # TODO: 可擴充台灣國定假日判斷
    # 例如從 https://www.twse.com.tw/zh/holidaySchedule/holidaySchedule 取得
    
    return True


# =====================
# 執行子腳本
# =====================
def run_script(script_name: str, args: list = None) -> tuple:
    """
    執行 Python 腳本
    
    Returns:
        tuple: (success: bool, output: str)
    """
    cmd = [sys.executable, os.path.join(BASE_DIR, script_name)]
    if args:
        cmd.extend(args)
        
    print(f"🔄 執行: {script_name} {' '.join(args or [])}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=1800,  # 30 分鐘超時 (夏普掃描需要較長時間)
            cwd=BASE_DIR
        )
        
        if result.returncode == 0:
            print(f"✅ {script_name} 執行成功")
            return True, result.stdout
        else:
            print(f"❌ {script_name} 執行失敗")
            print(f"   錯誤: {result.stderr[:500]}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} 執行超時")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ {script_name} 執行錯誤: {e}")
        return False, str(e)


# =====================
# 解析掃描報告
# =====================
def parse_sharpe_report(output: str) -> str:
    """
    從 scan_market.py 輸出中擷取關鍵資訊
    """
    lines = output.split('\n')
    result_lines = []
    capture = False
    
    for line in lines:
        if '夏普比率' in line or 'Sharpe' in line or '排名' in line:
            capture = True
        if capture:
            result_lines.append(line)
            if len(result_lines) >= 10:  # 最多取 10 行
                break
                
    return '\n'.join(result_lines) if result_lines else "無夏普報告"


def parse_signal_report(output: str) -> str:
    """
    從 signal_scanner.py 輸出擷取關鍵資訊
    """
    lines = output.split('\n')
    result_lines = []
    
    for line in lines:
        # 擷取包含訊號的行
        if any(kw in line for kw in ['買入', 'BUY', '訊號', '連買', '突破', 'Top']):
            result_lines.append(line)
            if len(result_lines) >= 15:
                break
                
    return '\n'.join(result_lines) if result_lines else "無訊號報告"


# =====================
# 主流水線
# =====================
def run_pipeline(skip_download: bool = False, notify_channel: str = 'line'):
    """
    執行完整自動化流水線
    
    Args:
        skip_download: 是否跳過下載 (測試用)
        notify_channel: 通知通道 ('line' 或 'telegram')
    """
    today = datetime.now()
    date_str = today.strftime('%Y-%m-%d')
    
    print("=" * 50)
    print(f"📅 量化自動化流水線 - {date_str}")
    print("=" * 50)
    
    # 1. 交易日判斷
    if not is_trading_day(today):
        print(f"🛑 今日 ({date_str}) 非交易日，跳過執行")
        return
        
    reports = []
    errors = []
    
    # 2. 下載資料
    if not skip_download:
        success, output = run_script('downloader_tw.py')
        if not success:
            errors.append("資料下載失敗")
    else:
        print("⏭️ 跳過下載 (skip_download=True)")
    
    # 3. 夏普掃描
    success, output = run_script('scan_market.py')
    if success:
        sharpe_summary = parse_sharpe_report(output)
        reports.append(f"🔥 夏普掃描結果:\n{sharpe_summary}")
    else:
        errors.append("夏普掃描失敗")
    
    # 4. 訊號掃描
    success, output = run_script('signal_scanner.py')
    if success:
        signal_summary = parse_signal_report(output)
        reports.append(f"📈 訊號掃描結果:\n{signal_summary}")
    else:
        errors.append("訊號掃描失敗")
    
    # 5. 組合通知訊息
    message = f"""
📊 量化掃描報告 ({date_str})
{'=' * 30}
"""
    
    for report in reports:
        message += f"\n{report}\n"
        
    if errors:
        message += f"\n⚠️ 執行錯誤:\n" + '\n'.join(f"  - {e}" for e in errors)
        
    message += f"\n{'=' * 30}\n⏱ 執行時間: {datetime.now().strftime('%H:%M')}"
    
    # 6. 發送通知
    print("\n📤 發送通知...")
    notify(message, channel=notify_channel)
    
    print("\n✅ 流水線執行完成！")


# =====================
# CLI 入口
# =====================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='量化自動化流水線')
    parser.add_argument('--skip-download', action='store_true', help='跳過下載步驟')
    parser.add_argument('--channel', default='line', choices=['line', 'telegram'], help='通知通道')
    parser.add_argument('--force', action='store_true', help='強制執行 (即使非交易日)')
    
    args = parser.parse_args()
    
    # 如果強制執行，暫時覆寫交易日判斷
    if args.force:
        print("⚠️ 強制執行模式，忽略交易日判斷")
        # Monkey patch
        is_trading_day = lambda d=None: True
        
    run_pipeline(skip_download=args.skip_download, notify_channel=args.channel)
