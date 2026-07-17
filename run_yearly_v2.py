#!/usr/bin/env python3
"""A股可转债 MC定价 一年回测 v2 — 快速版"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
import httpx, yfinance as yf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import MCConfig
from pricer import CCBPricer, CCBParams

EP = "https://market.ft.tech/gateway/mcp"
H = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}

def init_mcp():
    c = httpx.Client(timeout=30)
    r = c.post(EP, json={"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"bt","version":"1"}},"id":1}, headers=H)
    sid = r.headers.get("mcp-session-id","")
    c.post(EP, json={"jsonrpc":"2.0","method":"notifications/initialized","id":2}, headers={**H,"mcp-session-id":sid})
    return c, sid

def mcp_call(c, sid, tool, args=None):
    r = c.post(EP, json={"jsonrpc":"2.0","method":"tools/call","params":{"name":tool,"arguments":args or {}},"id":int(time.time()*1000)%100000}, headers={**H,"mcp-session-id":sid})
    for line in r.text.split("\n"):
        if line.startswith("data:") and '"text"' in line:
            try:
                d = json.loads(line[5:].strip())
                for ct in d.get("result",{}).get("content",[]):
                    if ct.get("type")=="text": return ct["text"]
            except: pass
    return ""

def parse_md(text):
    rows=[]; hdr=None
    for l in text.split("\n"):
        if not l.startswith("|"): continue
        cols=[x.strip() for x in l.split("|")[1:-1]]
        if "---" in l: continue
        if hdr is None: hdr=cols; continue
        if hdr: rows.append(dict(zip(hdr,cols)))
    return rows


print("="*60)
print("A股可转债 MC定价 一年回测 v2")
print("="*60)

# Phase 1: Get bond list from MCP (1 call)
print("\n[1/3] 获取可转债列表...")
c, sid = init_mcp()
raw = mcp_call(c, sid, "ft_get_cb_lists_handler")
all_bonds = parse_md(raw)
print(f"  共 {len(all_bonds)} 只")

# Build stock_id → cb_id mapping, prepare yfinance tickers
yf_cb_tickers = []
yf_st_tickers = []
bond_map = {}  # cb_id -> {stock_id, cb_sym}
for b in all_bonds:
    cb_id = b["cb_id"]; stock_id = b["stock_id"]
    if cb_id.startswith("11"): cb_sym = f"{cb_id}.SS"
    elif cb_id.startswith("12"): cb_sym = f"{cb_id}.SZ"
    else: continue
    if stock_id.startswith("6"): st_sym = f"{stock_id}.SS"
    elif stock_id.startswith(("0","2","3")): st_sym = f"{stock_id}.SZ"
    else: continue
    yf_cb_tickers.append(cb_sym)
    yf_st_tickers.append(st_sym)
    bond_map[cb_id] = {"stock_id": stock_id, "cb_sym": cb_sym, "st_sym": st_sym}

print(f"  yfinance映射: {len(bond_map)} 只")

# Phase 2: Download 1 year of data from yfinance
print("\n[2/3] yfinance 下载历史行情 (2025-07 ~ 2026-07)...")
end = "2026-07-13"; start = "2025-07-13"

# Download CB prices
t0 = time.time()
cb_data = {}
for i in tqdm(range(0, len(yf_cb_tickers), 100), desc="转债行情"):
    batch = yf_cb_tickers[i:i+100]
    try:
        data = yf.download(batch, start=start, end=end, progress=False, auto_adjust=True)
        for ticker in batch:
            cb_id = list(bond_map.keys())[list(bond_map.values()).index({k:v for k,v in bond_map.items() if v["cb_sym"]==ticker}.get("cb_sym",None))] if ticker in [v["cb_sym"] for v in bond_map.values()] else None
            if len(batch)==1: closes = data.get("Close", pd.Series(dtype=float))
            else:
                try: closes = data["Close"][ticker]
                except: closes = pd.Series(dtype=float)
            if isinstance(closes, pd.Series) and len(closes.dropna()) >= 50:
                cb_data[ticker] = closes.dropna()
    except Exception as e: print(f"  batch err: {e}")

# Simpler: use dict comprehension
cb_data2 = {}
for i in tqdm(range(0, len(yf_cb_tickers), 100), desc="转债行情v2"):
    batch = yf_cb_tickers[i:i+100]
    batch_str = " ".join(batch)
    try:
        data = yf.download(batch_str, start=start, end=end, progress=False, auto_adjust=True, group_by="ticker")
        if len(batch)==1:
            s = data["Close"].dropna()
            if len(s)>=50: cb_data2[batch[0]] = s
        else:
            for t in batch:
                try:
                    s = data[t]["Close"].dropna()
                    if len(s)>=50: cb_data2[t] = s
                except: pass
    except Exception as e: print(f"  err: {e}")

# Download stock prices
st_data = {}
for i in tqdm(range(0, len(yf_st_tickers), 100), desc="正股行情"):
    batch = yf_st_tickers[i:i+100]
    try:
        data = yf.download(" ".join(batch), start=start, end=end, progress=False, auto_adjust=True, group_by="ticker")
        if len(batch)==1:
            s = data["Close"].dropna()
            if len(s)>=50: st_data[batch[0]] = s
        else:
            for t in batch:
                try:
                    s = data[t]["Close"].dropna()
                    if len(s)>=50: st_data[t] = s
                except: pass
    except: pass

print(f"  CB: {len(cb_data2)} 只有效行情, 正股: {len(st_data)} 只有效行情")

# Filter bonds with both data
valid = []
for cb_id, bm in bond_map.items():
    if bm["cb_sym"] in cb_data2 and bm["st_sym"] in st_data:
        valid.append(cb_id)

print(f"  完整数据: {len(valid)} 只")
if len(valid) < 10:
    print("  数据不足!"); sys.exit(1)

# Phase 3: Backtest loop
print(f"\n[3/3] MC定价回测...")

config = MCConfig(n_paths=50, n_days=252, use_gpu=True)
pricer = CCBPricer(config)

# Build aligned DataFrame
dates = sorted(set(cb_data2[list(valid)[0]].index.tolist()))
valid_dates = dates[252:]  # Need 252 days for vol

portfolio = [1.0]
daily_rets = []
prev_holdings = set()
tc = 0.001

for di, today in enumerate(tqdm(valid_dates, desc="回测")):
    if di + 1 >= len(valid_dates): break
    tomorrow = valid_dates[di+1]
    
    # Compute discounts for today
    discounts = []
    for cb_id in valid:
        bm = bond_map[cb_id]
        cb_s = cb_data2[bm["cb_sym"]]
        st_s = st_data[bm["st_sym"]]
        
        if today not in cb_s.index or today not in st_s.index: continue
        
        stock_p = float(st_s.loc[today])
        cb_p = float(cb_s.loc[today])
        if stock_p <= 0 or cb_p <= 0: continue
        
        # Historical vol
        hist = st_s.loc[:today].iloc[-252:]
        if len(hist) >= 20:
            vol = float(np.std(np.diff(np.log(hist.values))) * np.sqrt(252))
            vol = max(0.05, min(vol, 1.5))
        else: vol = 0.3
        
        # Conversion price from ratio of stock/cb
        conv_p = stock_p * 0.95  # Approximate
        
        ccb = CCBParams(name=cb_id, ticker=cb_id, stock_ticker=bm["stock_id"],
                       face_value=100.0, conversion_price=conv_p,
                       days_to_maturity=252*2, call_trigger_pct=1.30,
                       put_trigger_pct=0.70, down_trigger_pct=0.85,
                       put_price=100.0, redemption_price=108.0,
                       market_price=cb_p, stock_price=stock_p,
                       volatility=vol, conversion_start_day=0, industry="通用")
        try:
            mc_price, _ = pricer.price_single(ccb, n_paths=50, step_days=63, seed=42)
            disc = (mc_price - cb_p) / cb_p
            discounts.append({"cb_id": cb_id, "cb_p": cb_p, "stock_p": stock_p, "disc": disc, "cb_s": cb_s})
        except: continue
    
    if len(discounts) < 10: continue
    
    # Top 10
    discounts.sort(key=lambda x: x["disc"], reverse=True)
    top10 = discounts[:10]
    top10_ids = set(d["cb_id"] for d in top10)
    
    # Next-day returns
    pos_rets = []
    for d in top10:
        if tomorrow in d["cb_s"].index:
            tmr = float(d["cb_s"].loc[tomorrow])
            if tmr > 0:
                pos_rets.append((tmr - d["cb_p"]) / d["cb_p"])
    
    if pos_rets:
        port_ret = np.mean(pos_rets)
        port_ret -= tc * len(top10_ids - prev_holdings) / 10
        prev_holdings = top10_ids
    else:
        port_ret = 0
    
    daily_rets.append(port_ret)
    portfolio.append(portfolio[-1] * (1 + port_ret))

# Results
ret_s = pd.Series(daily_rets)
cum_ret = portfolio[-1] - 1
ann_ret = (1+cum_ret)**(252/len(daily_rets)) - 1
ann_vol = ret_s.std() * np.sqrt(252)
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
cum_s = (1+ret_s).cumprod()
max_dd = (cum_s / cum_s.expanding().max() - 1).min()
win_rate = (ret_s > 0).mean()

# Benchmark: equal weight all
bench_rets = []
for di, today in enumerate(valid_dates[:len(daily_rets)]):
    if di+1 >= len(valid_dates): break
    tm = valid_dates[di+1]
    day_rets = []
    for cb_id in list(valid)[:50]:
        s = cb_data2[bond_map[cb_id]["cb_sym"]]
        if today in s.index and tm in s.index:
            p_t = float(s.loc[today]); p_tm = float(s.loc[tm])
            if p_t > 0: day_rets.append((p_tm-p_t)/p_t)
    bench_rets.append(np.mean(day_rets) if day_rets else 0)
bench_cum = (1+pd.Series(bench_rets)).prod() - 1

print("\n" + "=" * 60)
print("A股可转债 MC定价 一年回测 结果")
print("=" * 60)
print(f"  {'指标':<20} {'LSM策略':<15} {'等权基准':<15} {'差异':<15}")
print(f"  {'累计收益':<20} {cum_ret*100:>14.2f}% {bench_cum*100:>14.2f}% {cum_ret*100-bench_cum*100:>+14.2f}%")
print(f"  {'年化收益':<20} {ann_ret*100:>14.2f}% {'—':>15}")
print(f"  {'年化波动':<20} {ann_vol*100:>14.2f}% {'—':>15}")
print(f"  {'夏普比率':<20} {sharpe:>14.2f} {'—':>15}")
print(f"  {'最大回撤':<20} {max_dd*100:>14.2f}% {'—':>15}")
print(f"  {'日胜率':<20} {win_rate*100:>14.1f}% {'—':>15}")
print(f"\n  转债: {len(valid)}只, 交易日: {len(daily_rets)}天, MC定价: 100条路径")

# Save
out = Path(__file__).parent / "output_backtest"
out.mkdir(exist_ok=True)
ts = time.strftime("%Y%m%d_%H%M%S")
pd.DataFrame({"daily_return": daily_rets, "cumulative": portfolio[1:]}).to_csv(out/f"yearly_{ts}.csv", index=False)
print(f"\n结果: {out}/yearly_{ts}.csv")
