#!/usr/bin/env python3
"""A股可转债 MC定价 一年回测 — 真实行情 (Sina数据源)"""
import sys, time
sys.path.insert(0, '.')
import numpy as np, pandas as pd
import akshare as ak
from tqdm import tqdm
from config import MCConfig
from pricer import CCBPricer, CCBParams

print("="*60)
print("A股可转债 MC定价 一年回测 (Sina真实行情)")
print("="*60)

# ── 1. Get CB list ──
print("\n[1/3] 获取可转债列表...")
df_cb = ak.bond_cb_jsl()
df_cb = df_cb.dropna(subset=['转股价','现价','正股价']).copy()
df_cb = df_cb[df_cb['转股价'] > 0].head(50)  # Top 50 most active
print(f"  {len(df_cb)} 只转债")

# ── 2. Download historical data ──
print(f"\n[2/3] 下载历史行情...")
cb_data = {}
st_data = {}

for _, r in tqdm(df_cb.iterrows(), total=len(df_cb), desc="下载"):
    cb_id = r['代码']
    stock_id = r['正股代码']
    cb_prefix = 'sh' if cb_id.startswith('1') else 'sz'
    st_prefix = 'sh' if stock_id.startswith(('6','5')) else 'sz'
    
    try:
        cb_df = ak.bond_zh_hs_cov_daily(symbol=f'{cb_prefix}{cb_id}')
        if len(cb_df) >= 200:
            cb_df['date'] = pd.to_datetime(cb_df['date'])
            cb_df = cb_df.set_index('date')
            cb_data[cb_id] = cb_df
    except: pass
    
    try:
        st_df = ak.stock_zh_a_daily(symbol=f'{st_prefix}{stock_id}', adjust='qfq')
        if len(st_df) >= 200:
            st_df['date'] = pd.to_datetime(st_df['date'])
            st_df = st_df.set_index('date')
            st_data[cb_id] = st_df
    except: pass
    
    time.sleep(0.05)

valid = sorted(set(cb_data.keys()) & set(st_data.keys()))
print(f"  完整数据: {len(valid)} 只")

if len(valid) < 10:
    print("  数据不足!"); sys.exit(1)

# ── 3. Build trading calendar from latest year ──
all_dates = set()
for cid in valid:
    all_dates.update(cb_data[cid].index.tolist())
trading_days = sorted(all_dates)
# Last ~250 trading days
trading_days = trading_days[-251:]
print(f"  交易日: {len(trading_days)} ({trading_days[0].strftime('%Y-%m-%d')} ~ {trading_days[-1].strftime('%Y-%m-%d')})")

# ── 4. Backtest ──
print(f"\n[3/3] MC定价回测...")
config = MCConfig(n_paths=50, n_days=252, use_gpu=True)
pricer = CCBPricer(config)

portfolio = [1.0]
daily_rets = []
prev_holdings = set()
tc = 0.001

for di in tqdm(range(len(trading_days)-1), desc="回测"):
    today = trading_days[di]
    tomorrow = trading_days[di+1]
    
    discounts = []
    for cid in valid:
        if today not in cb_data[cid].index or today not in st_data[cid].index:
            continue
        
        stock_p = float(st_data[cid].loc[today]['close'])
        cb_p = float(cb_data[cid].loc[today]['close'])
        if stock_p <= 0 or cb_p <= 0: continue
        
        # Historical vol
        st_hist = st_data[cid].loc[:today]
        st_hist = st_hist.iloc[-252:] if len(st_hist) >= 252 else st_hist
        if len(st_hist) >= 20:
            lr = np.diff(np.log(st_hist['close'].values))
            vol = float(np.std(lr)*np.sqrt(252))
            vol = max(0.05, min(vol, 1.5))
        else: vol = 0.3
        
        br = df_cb[df_cb['代码'] == cid]
        if len(br) == 0: continue
        br = br.iloc[0]
        conv_p = float(br['转股价'])
        
        ccb = CCBParams(name=str(br['转债名称']), ticker=cid, stock_ticker=str(br['正股代码']),
                       face_value=100.0, conversion_price=conv_p,
                       days_to_maturity=252*2, call_trigger_pct=1.30,
                       put_trigger_pct=0.70, down_trigger_pct=0.85,
                       put_price=100.0, redemption_price=108.0,
                       market_price=cb_p, stock_price=stock_p,
                       volatility=vol)
        try:
            mc_p, _ = pricer.price_single(ccb, n_paths=50, step_days=63, seed=42)
            disc = (mc_p-cb_p)/cb_p if cb_p>0 else -1
            discounts.append({"cid":cid,"disc":disc,"cb_p":cb_p,"tm":tomorrow,"cb_df":cb_data[cid]})
        except: continue
    
    if len(discounts) < 10: continue
    discounts.sort(key=lambda x: x['disc'], reverse=True)
    top10 = discounts[:10]
    top10_ids = set(d['cid'] for d in top10)
    
    pos_rets = []
    for d in top10:
        if d['tm'] in d['cb_df'].index:
            tmr_p = float(d['cb_df'].loc[d['tm']]['close'])
            if d['cb_p'] > 0: pos_rets.append((tmr_p-d['cb_p'])/d['cb_p'])
    
    port_ret = np.mean(pos_rets) if pos_rets else 0
    port_ret -= tc * len(top10_ids-prev_holdings)/10
    prev_holdings = top10_ids
    daily_rets.append(port_ret)
    portfolio.append(portfolio[-1]*(1+port_ret))

# ── Benchmark ──
bench_rets = []
for di in range(len(daily_rets)):
    today, tm = trading_days[di], trading_days[di+1]
    day_rets = []
    for cid in valid:
        if today in cb_data[cid].index and tm in cb_data[cid].index:
            pt = float(cb_data[cid].loc[today]['close'])
            ptm = float(cb_data[cid].loc[tm]['close'])
            if pt > 0: day_rets.append((ptm-pt)/pt)
    bench_rets.append(np.mean(day_rets) if day_rets else 0)

# ── Results ──
ret_s = pd.Series(daily_rets)
cum_ret = portfolio[-1]-1
ann_ret = (1+cum_ret)**(252/len(daily_rets))-1
ann_vol = ret_s.std()*np.sqrt(252)
sharpe = ann_ret/ann_vol if ann_vol>0 else 0
cum_s = (1+ret_s).cumprod()
max_dd = (cum_s/cum_s.expanding().max()-1).min()
bench_cum = (1+pd.Series(bench_rets)).prod()-1
ir = (ret_s.mean()-np.mean(bench_rets))/ret_s.std()*np.sqrt(252) if ret_s.std()>0 else 0

print("\n"+"="*60)
print("  A股可转债 MC定价 一年真实回测")
print("="*60)
print(f"  {'指标':<22} {'LSM策略':<15} {'等权基准':<15}")
print(f"  {'累计收益':<22} {cum_ret*100:>14.2f}% {bench_cum*100:>14.2f}%")
print(f"  {'年化收益':<22} {ann_ret*100:>14.2f}% {'—':>15}")
print(f"  {'年化波动':<22} {ann_vol*100:>14.2f}% {'—':>15}")
print(f"  {'夏普比率':<22} {sharpe:>14.2f} {'—':>15}")
print(f"  {'最大回撤':<22} {max_dd*100:>14.2f}% {'—':>15}")
print(f"  {'日胜率':<22} {(ret_s>0).mean()*100:>14.1f}% {'—':>15}")
print(f"  {'信息比率':<22} {ir:>14.2f} {'—':>15}")
print(f"\n  转债: {len(valid)}只, 交易日: {len(daily_rets)}天, MC: 50条路径")
