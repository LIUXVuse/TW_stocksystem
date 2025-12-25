# -*- coding: utf-8 -*-
"""
通知服務模組
支援 Line Notify 與 Telegram Bot
"""
import os
import requests

# =====================
# Line Notify
# =====================
def send_line_notify(message: str, token: str = None) -> bool:
    """
    透過 Line Notify 發送訊息
    
    Args:
        message: 訊息內容
        token: Line Notify Token (若未提供則從環境變數 LINE_NOTIFY_TOKEN 讀取)
        
    Returns:
        bool: 是否成功發送
    """
    token = token or os.environ.get('LINE_NOTIFY_TOKEN')
    
    if not token:
        print("⚠️ 未設定 LINE_NOTIFY_TOKEN 環境變數")
        return False
        
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {token}'}
    data = {'message': message}
    
    try:
        response = requests.post(url, headers=headers, data=data, timeout=10)
        if response.status_code == 200:
            print("✅ Line Notify 發送成功")
            return True
        else:
            print(f"❌ Line Notify 發送失敗: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Line Notify 發送錯誤: {e}")
        return False


# =====================
# Telegram Bot
# =====================
def send_telegram(message: str, bot_token: str = None, chat_id: str = None) -> bool:
    """
    透過 Telegram Bot 發送訊息
    
    Args:
        message: 訊息內容 (支援 Markdown)
        bot_token: Bot Token (若未提供則從環境變數 TELEGRAM_BOT_TOKEN 讀取)
        chat_id: Chat ID (若未提供則從環境變數 TELEGRAM_CHAT_ID 讀取)
        
    Returns:
        bool: 是否成功發送
    """
    bot_token = bot_token or os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = chat_id or os.environ.get('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("⚠️ 未設定 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID 環境變數")
        return False
        
    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    data = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            print("✅ Telegram 發送成功")
            return True
        else:
            print(f"❌ Telegram 發送失敗: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Telegram 發送錯誤: {e}")
        return False


# =====================
# 統一介面
# =====================
def notify(message: str, channel: str = 'line') -> bool:
    """
    統一通知介面
    
    Args:
        message: 訊息內容
        channel: 通道 ('line' 或 'telegram')
        
    Returns:
        bool: 是否成功發送
    """
    if channel == 'line':
        return send_line_notify(message)
    elif channel == 'telegram':
        return send_telegram(message)
    else:
        print(f"⚠️ 不支援的通知通道: {channel}")
        return False


# =====================
# 測試區
# =====================
if __name__ == '__main__':
    import sys
    
    test_message = """
📊 量化掃描報告 (測試)
-------------------------------
🔥 夏普 Top 3:
  1. 2454 聯發科 (Sharpe: 2.1)
  2. 2330 台積電 (Sharpe: 1.8)
  3. 2317 鴻海 (Sharpe: 1.5)
-------------------------------
⏱ 這是測試訊息
"""
    
    if len(sys.argv) > 1:
        channel = sys.argv[1]
        print(f"🔔 測試 {channel} 通知...")
        notify(test_message, channel=channel)
    else:
        print("💡 用法: python notifier.py [line|telegram]")
        print("   需先設定環境變數:")
        print("   - LINE_NOTIFY_TOKEN")
        print("   - TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")
