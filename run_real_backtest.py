#!/usr/bin/env python3
"""A股可转债 MC定价 一年回测 — 真实行情版 (akshare + 东方财富API)"""
import sys, time
sys.path.insert(0, '.')
import numpy as np, pandas as pd
import requests, akshare as ak
from tqdm import tqdm
from config import MCConfig
from pricer import CCBPricer, CCBParams

print("="*60)
print("A股可转债 MC定价 一年回测 (真实行情)")
print("="*60)

# ── 1. Get live CB list from 集思录 ──
print("\n[1/3] 获取可转债实时数据 (集思录)...")
df_cb = ak.bond_cb_jsl()
print(f"  {len(df_cb)} 只转债")

# Filter active bonds with necessary data
df_cb = df_cb.dropna(subset=['转股价','现价','正股价']).copy()
df_cb = df_cb[df_cb['转股价'] > 0]

print(f"  {len(df_cb)} 只有效数据")
for _, r in df_cb.head(5).iterrows():
    print(f"  {r['代码']} {r['转债名称'][:15]:15s} 转股价={r['转股价']:.2f} 现价={r['现价']:.2f} 正股价={r['正股价']:.2f}")

# ── 2. Download 1 year historical OHLC ──
print(f"\n[2/3] 下载历史行情 (东方财富API)...")

def em_kline(secid, beg='20250701', end='20260713', klt='101', fqt='1'):
    """东方财富K线API: secid格式 '1.601166'(沪市) 或 '0.300498'(深市)"""
    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    r = requests.get(url, params={
        'secid': secid, 'klt': klt, 'fqt': fqt,
        'beg': beg, 'end': end,
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61'
    }, timeout=30)
    data = r.json().get('data', {})
    if not data or not data.get('klines'): return None
    records = []
    for k in data['klines']:
        parts = k.split(',')
        records.append({
            'date': parts[0], 'open': float(parts[1]), 'close': float(parts[2]),
            'high': float(parts[3]), 'low': float(parts[4]), 'volume': float(parts[5]),
        })
    return pd.DataFrame(records).set_index('date')

# Build secid mapping: SH=1.xxx, SZ=0.xxx
def to_secid(code):
    if code.startswith(('6','5','1')): return f'1.{code}'  # 上海
    else: return f'0.{code}'  # 深圳

cb_data = {}   # cb_id -> DataFrame of daily CB prices
st_data = {}   # cb_id -> DataFrame of daily stock prices

for _, r in tqdm(df_cb.iterrows(), total=len(df_cb), desc="下载行情"):
    cb_id = r['代码']
    stock_id = r['正股代码']
    
    # CB K-line
    cb_secid = to_secid(cb_id)
    cb_df = em_kline(cb_secid)
    if cb_df is not None and len(cb_df) >= 60:
        cb_data[cb_id] = cb_df
    
    # Stock K-line
    st_secid = to_secid(stock_id)
    st_df = em_kline(st_secid)
    if st_df is not None and len(st_df) >= 60:
        st_data[cb_id] = st_df
    
    time.sleep(0.1)

# Intersection: bonds with both CB and stock data
valid_ids = set(cb_data.keys()) & set(st_data.keys())
print(f"  有完整行情数据: {len(valid_ids)} 只")

if len(valid_ids) < 10:
    print("  数据不足!"); sys.exit(1)

# ── 3. Backtest ──
print(f"\n[3/3] MC定价回测 ({len(valid_ids)} 只转债)...")

# Build unified trading calendar
all_dates = set()
for cid in valid_ids:
    all_dates.update(cb_data[cid].index.tolist())
trading_days = sorted(all_dates)
trading_days = trading_days[252:]  # need 252 for vol
print(f"  有效交易日: {len(trading_days)}")

config = MCConfig(n_paths=50, n_days=252, use_gpu=True)
pricer = CCBPricer(config)

portfolio = [1.0]
daily_rets = []
prev_holdings = set()
tc = 0.001

for di, today in enumerate(tqdm(trading_days[:251], desc="回测")):
    if di + 1 >= len(trading_days): break
    tomorrow = trading_days[di+1]
    
    # Price all bonds
    discounts = []
    for cid in valid_ids:
        if today not in cb_data[cid].index or today not in st_data[cid].index:
            continue
        
        cb_row = cb_data[cid].loc[today]
        st_row = st_data[cid].loc[today]
        
        stock_p = float(st_row['close'])
        cb_p = float(cb_row['close'])
        if stock_p <= 0 or cb_p <= 0: continue
        
        # Vol from stock history
        st_hist = st_data[cid].loc[:today]
        if len(st_hist) >= 252:
            st_hist = st_hist.iloc[-252:]
        if len(st_hist) >= 20:
            lr = np.diff(np.log(st_hist['close'].values))
            vol = float(np.std(lr) * np.sqrt(252))
            vol = max(0.05, min(vol, 1.5))
        else: vol = 0.3
        
        # Get bond params
        bond_row = df_cb[df_cb['代码'] == cid]
        if len(bond_row) == 0: continue
        br = bond_row.iloc[0]
        conv_p = float(br['转股价'])
        
        ccb = CCBParams(name=str(br['转债名称']), ticker=cid, stock_ticker=str(br['正股代码']),
                       face_value=100.0, conversion_price=conv_p,
                       days_to_maturity=252*2, call_trigger_pct=1.30,
                       put_trigger_pct=0.70, down_trigger_pct=0.85,
                       put_price=100.0, redemption_price=108.0,
                       market_price=cb_p, stock_price=stock_p,
                       volatility=vol, conversion_start_day=0)
        try:
            mc_p, _ = pricer.price_single(ccb, n_paths=50, step_days=63, seed=42)
            disc = (mc_p - cb_p) / cb_p if cb_p > 0 else -1
            discounts.append({"cid": cid, "disc": disc, "cb_p": cb_p, "cb_df": cb_data[cid], "tm": tomorrow})
        except: continue
    
    if len(discounts) < 10: continue
    discounts.sort(key=lambda x: x['disc'], reverse=True)
    top10 = discounts[:10]
    top10_ids = set(d['cid'] for d in top10)
    
    # Next-day returns
    pos_rets = []
    for d in top10:
        if d['tm'] in d['cb_df'].index:
            tmr_p = float(d['cb_df'].loc[d['tm']]['close'])
            if d['cb_p'] > 0: pos_rets.append((tmr_p - d['cb_p']) / d['cb_p'])
    
    if pos_rets:
        port_ret = np.mean(pos_rets)
        port_ret -= tc * len(top10_ids - prev_holdings) / 10
        prev_holdings = top10_ids
    else:
        port_ret = 0
    
    daily_rets.append(port_ret)
    portfolio.append(portfolio[-1] * (1 + port_ret))

# ── Benchmark: equal-weight all ──
bench_rets = []
for di in range(len(daily_rets)):
    today = trading_days[di]; tm = trading_days[di+1]
    day_rets = []
    for cid in list(valid_ids)[:30]:
        if today in cb_data[cid].index and tm in cb_data[cid].index:
            pt = float(cb_data[cid].loc[today]['close'])
            ptm = float(cb_data[cid].loc[tm]['close'])
            if pt > 0: day_rets.append((ptm-pt)/pt)
    bench_rets.append(np.mean(day_rets) if day_rets else 0)

# ── Results ──
ret_s = pd.Series(daily_rets)
cum_ret = portfolio[-1] - 1
ann_ret = (1+cum_ret)**(252/len(daily_rets))-1
ann_vol = ret_s.std()*np.sqrt(252)
sharpe = ann_ret/ann_vol if ann_vol>0 else 0
cum_s = (1+ret_s).cumprod()
max_dd = (cum_s/cum_s.expanding().max()-1).min()
win = (ret_s>0).mean()
bench_cum = (1+pd.Series(bench_rets)).prod()-1
ir = (ret_s.mean()-np.mean(bench_rets))/ret_s.std()*np.sqrt(252)

print("\n"+"="*60)
print("  A股可转债 MC定价 一年真实回测")
print("="*60)
print(f"  {'指标':<22} {'LSM策略':<15} {'等权基准':<15}")
print(f"  {'累计收益':<22} {cum_ret*100:>14.2f}% {bench_cum*100:>14.2f}%")
print(f"  {'年化收益':<22} {ann_ret*100:>14.2f}% {'—':>15}")
print(f"  {'年化波动':<22} {ann_vol*100:>14.2f}% {'—':>15}")
print(f"  {'夏普比率':<22} {sharpe:>14.2f} {'—':>15}")
print(f"  {'最大回撤':<22} {max_dd*100:>14.2f}% {'—':>15}")
print(f"  {'日胜率':<22} {win*100:>14.1f}% {'—':>15}")
print(f"  {'信息比率':<22} {ir:>14.2f} {'—':>15}")
print(f"\n  转债: {len(valid_ids)}只, 交易日: {len(daily_rets)}天")
