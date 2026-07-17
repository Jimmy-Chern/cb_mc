"""
QMT客户端 — 从GitHub Gist拉取信号自动交易
在Windows笔记本QMT的Python环境中运行
每天14:50自动执行
"""
import requests, json, time

# ===== 你的Gist地址（公开可读，无需认证）=====
GIST_URL = "https://gist.githubusercontent.com/Jimmy-Chern/4f3d9550b2c5e874019272b025a58408/raw/cb_signal.json"

def get_signal():
    try:
        resp = requests.get(GIST_URL, timeout=15)
        data = resp.json()
        print(f"[{data['date']} {data['time']}] 信号: {data['n_positions']}只")
        for i, p in enumerate(data['positions']):
            print(f"  {i+1}. {p['code']} {p['name']} disc={p['discount']:+.1f}%")
        return data
    except Exception as e:
        print(f"信号获取失败: {e}")
        return None

def rebalance(signal):
    from xtquant import xtdata, xttrader
    
    if not signal: return
    target = set(signal['buy'])
    
    # 查当前持仓（转债代码11/12开头）
    acc = xttrader.query_stock_asset(0)
    current_pos = {}
    for pos in xttrader.query_stock_positions(0):
        if pos.stock_code.startswith(('11','12')):
            current_pos[pos.stock_code] = pos.volume
    
    current = set(current_pos.keys())
    to_sell = current - target
    to_buy  = target - current
    
    # 卖
    for code in to_sell:
        vol = current_pos.get(code, 0)
        if vol > 0:
            print(f"  卖出 {code} x{vol}")
            xttrader.sell(acc.account_id, code, volume=vol, price_type=5)
    
    # 买
    if to_buy:
        cash = acc.cash * 0.95
        per = cash / len(to_buy)
        for code in to_buy:
            price = xtdata.get_market_data('close', code, period='1d')
            if price and price > 0:
                vol = int(per / price / 10) * 10
                if vol >= 10:
                    print(f"  买入 {code} x{vol}")
                    xttrader.buy(acc.account_id, code, volume=vol, price_type=5)

# 每天14:50点执行
import schedule, datetime
print("QMT可转债MC策略客户端已启动")
print(f"信号源: {GIST_URL}")
print("每天14:50自动调仓")

schedule.every().day.at("14:50").do(lambda: rebalance(get_signal()))

while True:
    schedule.run_pending()
    time.sleep(30)
