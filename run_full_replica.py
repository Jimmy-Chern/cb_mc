#!/usr/bin/env python3
"""
100%论文复刻 — 309只集思录转债 + 缓存历史数据 + 5000路径MC定价
"""
import sys, time, os, pickle, torch
sys.path.insert(0, '/home/xujiayang2/chenjunming/cb_mc')
import numpy as np, pandas as pd
import akshare as ak
from pathlib import Path
from tqdm import tqdm
from config import MCConfig
from pricer import CCBPricer, CCBParams

# ── Config ──
CACHE_DIR = Path.home() / 'chenjunming/quant/cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
JISILU_FILE = Path.home() / 'chenjunming/quant/jisilu.txt'
M = 5000          # MC paths
STEP = 63         # Exercise check days  
N_POS = 10        # Top N holdings
TC = 0.001        # 0.1% fee

print("="*60)
print(" 论文复刻: M=5000, 309只转债, 真实行情+缓存")
print("="*60)

# ── 1. Parse jisilu.txt ──
print("\n[1/4] 解析集思录数据...")
lines = [l.strip() for l in open(JISILU_FILE) if '\t' in l and l.split('\t')[0].isdigit()]
bonds_config = []
for l in lines:
    f = l.split('\t')
    if len(f) < 25: continue
    try:
        cb_id = f[0]
        stock_id = f[4]
        conv_price = float(f[9].replace(',',''))
        cb_price = float(f[2].replace(',',''))
        stock_price = float(f[6].replace(',',''))
        name = f[1]
        remain_yr = float(f[23]) if f[23].replace('.','').replace('-','').isdigit() else 3.0
        rating = f[14] if '会员' not in f[14] else 'AA'
        if conv_price > 0 and stock_price > 0 and cb_price > 0:
            bonds_config.append({
                'cb_id': cb_id, 'stock_id': stock_id, 'name': name,
                'conv_price': conv_price, 'cb_price': cb_price,
                'stock_price': stock_price, 'remain_yr': remain_yr, 'rating': rating
            })
    except: continue

print(f"  解析: {len(bonds_config)} 只有效转债 (共{len(lines)}行)")

# ── 2. Download + Cache historical data ──
print(f"\n[2/4] 下载历史行情 (缓存目录: {CACHE_DIR})...")

cb_cache = CACHE_DIR / 'cb_hist.pkl'
st_cache = CACHE_DIR / 'st_hist.pkl'

if cb_cache.exists() and st_cache.exists():
    print("  从缓存加载...")
    with open(cb_cache, 'rb') as f: cb_hist = pickle.load(f)
    with open(st_cache, 'rb') as f: st_hist = pickle.load(f)
    print(f"  缓存: {len(cb_hist)} CB, {len(st_hist)} stock")
else:
    cb_hist = {}; st_hist = {}
    for bc in tqdm(bonds_config, desc="下载行情"):
        cb_id, st_id = bc['cb_id'], bc['stock_id']
        cb_pfx = 'sh' if cb_id.startswith('11') else 'sz'
        st_pfx = 'sh' if st_id.startswith(('6','5')) else 'sz'
        
        # CB data
        if cb_id not in cb_hist:
            try:
                df = ak.bond_zh_hs_cov_daily(symbol=f'{cb_pfx}{cb_id}')
                df['date'] = pd.to_datetime(df['date'])
                cb_hist[cb_id] = df.set_index('date')
            except: pass
        time.sleep(0.02)
        
        # Stock data  
        if cb_id not in st_hist:
            try:
                df = ak.stock_zh_a_daily(symbol=f'{st_pfx}{st_id}', adjust='qfq')
                df['date'] = pd.to_datetime(df['date'])
                st_hist[cb_id] = df.set_index('date')
            except: pass
        time.sleep(0.02)
    
    # Save cache
    with open(cb_cache, 'wb') as f: pickle.dump(cb_hist, f)
    with open(st_cache, 'wb') as f: pickle.dump(st_hist, f)
    print(f"  已缓存: {len(cb_hist)} CB, {len(st_hist)} stock")

# Filter bonds with complete data
valid = sorted(set(cb_hist.keys()) & set(st_hist.keys()))
# Filter bonds that also have config params
valid = [v for v in valid if any(b['cb_id']==v for b in bonds_config)]
# Only keep bonds with >= 200 days of data
valid = [v for v in valid if len(cb_hist[v]) >= 200 and len(st_hist[v]) >= 200]
print(f"  有效: {len(valid)} 只 (>=200天数据)")

if len(valid) < 10:
    print("数据不足!"); sys.exit(1)

# ── 3. Build trading calendar ──
# 全周期: 最近250个交易日 (2025-2026)
all_dates = set()
for cid in valid:
    all_dates.update(cb_hist[cid].index.tolist())
trading_days = sorted(all_dates)[-251:]
print(f"  交易日: {len(trading_days)} ({trading_days[0].strftime('%Y%m%d')}~{trading_days[-1].strftime('%Y%m%d')})")

# ── 4. 5000-path MC backtest ──
print(f"[4/4] MC定价回测 (M={M}, {len(valid)}只, {len(trading_days)-1}天)...")
config = MCConfig(n_paths=M, n_days=756, use_gpu=True)
pricer = CCBPricer(config)

portfolio = [1.0]; daily_rets = []; prev = set()
start = time.time()

for di in tqdm(range(len(trading_days)-1), desc="回测"):
    today, tomorrow = trading_days[di], trading_days[di+1]
    
    discounts = []
    for cid in valid:
        if today not in cb_hist[cid].index or today not in st_hist[cid].index:
            continue
        st_p = float(st_hist[cid].loc[today]['close'])
        cb_p = float(cb_hist[cid].loc[today]['close'])
        if st_p <= 0 or cb_p <= 0: continue
        
        # Historical vol
        st_h = st_hist[cid].loc[:today].iloc[-252:]
        vol = 0.3
        if len(st_h) >= 20:
            lr = np.diff(np.log(st_h['close'].values))
            vol = float(np.std(lr)*np.sqrt(252))
            vol = max(0.05, min(vol, 1.5))
        
        bc = next((b for b in bonds_config if b['cb_id']==cid), None)
        if bc is None: continue
        conv_p = bc['conv_price']
        days_mat = max(252, int(bc['remain_yr']*252))
        
        ccb = CCBParams(name=bc['name'], ticker=cid, stock_ticker=bc['stock_id'],
                       face_value=100.0, conversion_price=conv_p,
                       days_to_maturity=days_mat, call_trigger_pct=1.30,
                       put_trigger_pct=0.70, down_trigger_pct=0.85,
                       put_price=100.0, redemption_price=108.0,
                       call_mc=15, call_nc=30, put_mp=30, put_np=30,
                       market_price=cb_p, stock_price=st_p, volatility=vol,
                       conversion_start_day=0)
        try:
            mc_p, _ = pricer.price_single(ccb, n_paths=M, step_days=STEP, seed=42)
            disc = (mc_p-cb_p)/cb_p if cb_p>0 else -1
            discounts.append({"cid":cid, "disc":disc, "cb_p":cb_p, "tm":tomorrow})
        except: continue
    
    if len(discounts) < 10: continue
    discounts.sort(key=lambda x: x['disc'], reverse=True)
    top = discounts[:N_POS]
    top_ids = set(d['cid'] for d in top)
    
    pos_rets = []
    for d in top:
        if d['tm'] in cb_hist[d['cid']].index:
            tmr_p = float(cb_hist[d['cid']].loc[d['tm']]['close'])
            if d['cb_p'] > 0: pos_rets.append((tmr_p-d['cb_p'])/d['cb_p'])
    
    port_ret = np.mean(pos_rets) if pos_rets else 0
    port_ret -= TC * len(top_ids-prev)/N_POS
    prev = top_ids
    daily_rets.append(port_ret)
    portfolio.append(portfolio[-1]*(1+port_ret))
    # torch.cuda.empty_cache() removed — too slow, 10x overhead

elapsed = time.time() - start

# Benchmark
bench_rets = []
for di in range(len(daily_rets)):
    today, tm = trading_days[di], trading_days[di+1]
    day_rets = [float(cb_hist[cid].loc[tm]['close'])/float(cb_hist[cid].loc[today]['close'])-1
                for cid in valid if today in cb_hist[cid].index and tm in cb_hist[cid].index
                if float(cb_hist[cid].loc[today]['close'])>0]
    bench_rets.append(np.mean(day_rets) if day_rets else 0)

# Results
ret_s = pd.Series(daily_rets)
cum_ret = portfolio[-1]-1
ann_ret = (1+cum_ret)**(252/len(daily_rets))-1
ann_vol = ret_s.std()*np.sqrt(252)
sharpe = ann_ret/ann_vol if ann_vol>0 else 0
max_dd = ((1+ret_s).cumprod()/(1+ret_s).cumprod().expanding().max()-1).min()
bench_cum = (1+pd.Series(bench_rets)).prod()-1
ir = (ret_s.mean()-np.mean(bench_rets))/ret_s.std()*np.sqrt(252) if ret_s.std()>0 else 0

print("\n"+"="*60)
print("  论文复刻结果 — arxiv:2409.06496 (A股版)")
print("="*60)
print(f"  {'参数':<24} {'本回测':<16} {'论文':<16}")
print(f"  {'MC路径 M':<24} {M:>16d} {'5000':>16}")
print(f"  {'转债数量':<24} {len(valid):>16d} {'487':>16}")
print(f"  {'交易日':<24} {len(daily_rets):>16d} {'118':>16}")
print(f"  {'耗时':<24} {elapsed/60:>15.1f}分 {'—':>16}")
print(f"  {'—':<24} {'—':<16} {'—':<16}")
print(f"  {'累计收益':<24} {cum_ret*100:>15.2f}% {'29.17%':>16}")
print(f"  {'年化收益':<24} {ann_ret*100:>15.2f}% {'—':>16}")
print(f"  {'年化波动':<24} {ann_vol*100:>15.2f}% {'—':>16}")
print(f"  {'夏普比率':<24} {sharpe:>15.2f} {'1.20':>16}")
print(f"  {'最大回撤':<24} {max_dd*100:>15.2f}% {'20.00%':>16}")
print(f"  {'日胜率':<24} {(ret_s>0).mean()*100:>15.1f}% {'—':>16}")
print(f"  {'信息比率':<24} {ir:>15.2f} {'—':>16}")
print(f"  {'等权基准':<24} {bench_cum*100:>15.2f}% {'3.55%':>16}")
print(f"\n  数据缓存: {CACHE_DIR}")
