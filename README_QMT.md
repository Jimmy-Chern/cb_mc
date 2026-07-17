#!/usr/bin/env python3
"""QMT可转债MC策略 — 使用说明书"""

# ═══════════════════════════════════════════════════════
# 一、系统概述
# ═══════════════════════════════════════════════════════
#
# 浙大A800服务器每天15:30自动运行MC定价(5000条路径)，
# 对全市场212只可转债排序，选出理论价最被低估的Top10，
# 加密后推送到GitHub Gist。
#
# 这台Windows笔记本上的QMT每天14:50从Gist拉取加密信号，
# 解密后自动调仓。

# ═══════════════════════════════════════════════════════
# 二、环境准备（在QMT的Python里执行）
# ═══════════════════════════════════════════════════════
import os
os.system("pip install requests schedule pycryptodome -q")

# ═══════════════════════════════════════════════════════
# 三、QMT客户端代码（复制以下全部到QMT策略里）
# ═══════════════════════════════════════════════════════

import requests, json, time, hashlib, base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# === 配置 ===
GIST_URL = "https://gist.githubusercontent.com/Jimmy-Chern/4f3d9550b2c5e874019272b025a58408/raw/cb_signal.json"
PASSWORD = "336018"  # 加密密码，与服务器保持一致
TRADE_TIME = "14:50"  # 每天调仓时间

def decrypt_signal(encrypted_text, password):
    """AES解密：服务器用相同密码加密，这里解密"""
    key = hashlib.sha256(password.encode()).digest()
    raw = base64.b64decode(encrypted_text)
    iv = raw[:16]
    ciphertext = raw[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(ciphertext), 16)
    return json.loads(plaintext.decode('utf-8'))

def get_signal():
    """从Gist拉取加密信号并解密"""
    try:
        resp = requests.get(GIST_URL, timeout=15)
        data = resp.json()
        if "encrypted" in data:
            # 加密格式：{"encrypted": "base64密文"}
            signal = decrypt_signal(data["encrypted"], PASSWORD)
        else:
            signal = data  # 兼容未加密模式
        print(f"[{signal['date']} {signal['time']}] 信号: {signal['n_positions']}只")
        for i, p in enumerate(signal['positions']):
            print(f"  {i+1}. {p['code']} {p['name']} MC={p['model_price']} MKT={p['market_price']} disc={p['discount']:+.1f}%")
        return signal['buy']
    except Exception as e:
        print(f"获取信号失败: {e}")
        return None

def rebalance(buy_list):
    """对比持仓，调仓至目标组合"""
    from xtquant import xtdata, xttrader
    
    if not buy_list: return
    target = set(buy_list)
    
    acc_id = "你的资金账号"  # ⚠️ 改成你的实际账号
    acc = xttrader.query_stock_asset(acc_id)
    current_pos = {
        pos.stock_code: pos.volume
        for pos in xttrader.query_stock_positions(acc_id)
        if pos.stock_code.startswith(('11','12'))
    }
    
    to_sell = set(current_pos.keys()) - target
    to_buy  = target - set(current_pos.keys())
    
    for code in to_sell:
        vol = current_pos[code]
        print(f"  卖出 {code} x{vol}")
        xttrader.sell(acc_id, code, volume=vol, price_type=5)
    
    if to_buy:
        cash = acc.cash * 0.95
        per = cash / len(to_buy)
        for code in to_buy:
            price = xtdata.get_market_data('close', code, period='1d')
            if price and price > 0:
                vol = int(per / price / 10) * 10
                if vol >= 10:
                    print(f"  买入 {code} x{vol}")
                    xttrader.buy(acc_id, code, volume=vol, price_type=5)

# ═══════════════════════════════════════════════════════
# 四、启动
# ═══════════════════════════════════════════════════════
import schedule
print("QMT可转债MC定价策略客户端")
print(f"信号源: Gist (加密)")
print(f"调仓时间: 每天{TRADE_TIME}")

schedule.every().day.at(TRADE_TIME).do(lambda: rebalance(get_signal()))
while True:
    schedule.run_pending()
    time.sleep(30)
