#!/usr/bin/env python3
"""
100%复刻论文 Liu(2025) arxiv:2409.06496 — 中国A股可转债版
参数: M=5000路径, 全市场转债, 真实Sina行情, step=63天, 多区间回归
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np, pandas as pd
import akshare as ak
from tqdm import tqdm
from config import MCConfig
from pricer import CCBPricer, CCBParams

print("="*60)
print(" 论文复刻: arxiv:2409.06496 — A股可转债版")
print(" M=5000 paths, 全市场, 真实行情")
print("="*60)

# ── Paper parameters ──
M = 5000          # MC paths
EXERCISE_STEP = 63  # Exercise check interval (days)
N_POSITIONS = 10    # Top 10 holdings
TC = 0.001          # 0.1% transaction cost
N_DAYS = 250        # ~1 year

# ── 1. Get bond universe ──
print(f"\n[1/4] 获取全市场可转债...")
df_cb = ak.bond_cb_jsl()
df_cb = df_cb.dropna(subset=['转股价','现价','正股价']).copy()
df_cb = df_cb[df_cb['转股价'] > 0]
print(f"  全市场: {len(df_cb)} 只转债")

# ── 2. Download real historical data ──
print(f"[2/4] 下载历史行情...")
cb_hist = {}  # cb_id -> DataFrame
st_hist = {}  # cb_id -> DataFrame

for _, r in tqdm(df_cb.iterrows(), total=len(df_cb), desc="下载"):
    cb_id, st_id = r['代码'], r['正股代码']
    cb_pre = 'sh' if cb_id.startswith('1') else 'sz'
    st_pre = 'sh' if st_id.startswith(('6','5')) else 'sz'
    
    try:
        cb_df = ak.bond_zh_hs_cov_daily(symbol=f'{cb_pre}{cb_id}')
        cb_df['date'] = pd.to_datetime(cb_df['date'])
        cb_hist[cb_id] = cb_df.set_index('date')
    except: pass
    
    try:
        st_df = ak.stock_zh_a_daily(symbol=f'{st_pre}{st_id}', adjust='qfq')
        st_df['date'] = pd.to_datetime(st_df['date'])
        st_hist[cb_id] = st_df.set_index('date')
    except: pass
    time.sleep(0.03)

valid = sorted(set(cb_hist.keys()) & set(st_hist.keys()))
print(f"  有效: {len(valid)} 只")

if len(valid) < 10:
    print("数据不足!"); sys.exit(1)

# ── 3. Build trading calendar ──
all_dates = set()
for cid in valid:
    all_dates.update(cb_hist[cid].index.tolist())
trading_days = sorted(all_dates)[-N_DAYS-1:]
print(f"[3/4] 交易日: {len(trading_days)} ({trading_days[0].strftime('%Y%m%d')}~{trading_days[-1].strftime('%Y%m%d')})")

# ── 4. Backtest ──
print(f"[4/4] MC定价回测 (M={M}, {len(valid)}只, {N_DAYS}天)...")
config = MCConfig(n_paths=M, n_days=756, use_gpu=True)
pricer = CCBPricer(config)

portfolio = [1.0]; daily_rets = []; prev = set()
start_time = time.time()

for di in tqdm(range(len(trading_days)-1), desc="回测"):
    today, tomorrow = trading_days[di], trading_days[di+1]
    
    discounts = []
    for cid in valid:
        if today not in cb_hist[cid].index or today not in st_hist[cid].index:
            continue
        stock_p = float(st_hist[cid].loc[today]['close'])
        cb_p = float(cb_hist[cid].loc[today]['close'])
        if stock_p <= 0 or cb_p <= 0: continue
        
        # Historical vol (paper: last 252 days)
        st_h = st_hist[cid].loc[:today].iloc[-252:]
        vol = 0.3
        if len(st_h) >= 20:
            lr = np.diff(np.log(st_h['close'].values))
            vol = float(np.std(lr)*np.sqrt(252))
            vol = max(0.05, min(vol, 1.5))
        
        br = df_cb[df_cb['代码']==cid]
        if len(br)==0: continue
        br = br.iloc[0]
        conv_p = float(br['转股价'])
        days_mat = max(252, 252*2)  # ~2 year maturity
        
        # Build CCB with paper parameters
        ccb = CCBParams(
            name=str(br['转债名称']), ticker=cid, stock_ticker=str(br['正股代码']),
            face_value=100.0, conversion_price=conv_p,
            days_to_maturity=days_mat,
            call_trigger_pct=1.30, put_trigger_pct=0.70,
            down_trigger_pct=0.85, put_price=100.0, redemption_price=108.0,
            call_mc=15, call_nc=30, put_mp=30, put_np=30,
            market_price=cb_p, stock_price=stock_p, volatility=vol,
            conversion_start_day=0)
        
        try:
            mc_p, _ = pricer.price_single(ccb, n_paths=M, step_days=EXERCISE_STEP, seed=42)
            disc = (mc_p-cb_p)/cb_p if cb_p>0 else -1
            discounts.append({"cid":cid, "disc":disc, "cb_p":cb_p, "tm":tomorrow})
        except: continue
    
    if len(discounts) < 10: continue
    
    # Top N by discount (paper: buy top 10 most undervalued)
    discounts.sort(key=lambda x: x['disc'], reverse=True)
    top = discounts[:N_POSITIONS]
    top_ids = set(d['cid'] for d in top)
    
    # Equal-weight next-day return
    pos_rets = []
    for d in top:
        if d['tm'] in cb_hist[d['cid']].index:
            tmr_p = float(cb_hist[d['cid']].loc[d['tm']]['close'])
            if d['cb_p'] > 0: pos_rets.append((tmr_p-d['cb_p'])/d['cb_p'])
    
    port_ret = np.mean(pos_rets) if pos_rets else 0
    port_ret -= TC * len(top_ids-prev)/N_POSITIONS  # 0.1% transaction cost
    prev = top_ids
    daily_rets.append(port_ret)
    portfolio.append(portfolio[-1]*(1+port_ret))

elapsed = time.time() - start_time

# ── Benchmark: equal-weight all bonds ──
bench_rets = []
for di in range(len(daily_rets)):
    today, tm = trading_days[di], trading_days[di+1]
    day_rets = [float(cb_hist[cid].loc[tm]['close'])/float(cb_hist[cid].loc[today]['close'])-1
                for cid in valid if today in cb_hist[cid].index and tm in cb_hist[cid].index
                if float(cb_hist[cid].loc[today]['close'])>0]
    bench_rets.append(np.mean(day_rets) if day_rets else 0)

# ── Results ──
ret_s = pd.Series(daily_rets)
cum_ret = portfolio[-1]-1
ann_ret = (1+cum_ret)**(252/len(daily_rets))-1
ann_vol = ret_s.std()*np.sqrt(252)
sharpe = ann_ret/ann_vol if ann_vol>0 else 0
cum_s = (1+ret_s).cumprod()
max_dd = (cum_s/cum_s.expanding().max()-1).min()
win = (ret_s>0).mean()
bench_cum = (1+pd.Series(bench_rets)).prod()-1
ir = (ret_s.mean()-np.mean(bench_rets))/ret_s.std()*np.sqrt(252) if ret_s.std()>0 else 0

print("\n"+"="*60)
print("  论文复刻结果 — arxiv:2409.06496")
print("="*60)
print(f"  {'参数/指标':<24} {'本回测':<16} {'论文 Liu(2025)':<16}")
print(f"  {'MC路径数 M':<24} {M:>16d} {'5000':>16}")
print(f"  {'转债数量':<24} {len(valid):>16d} {'487':>16}")
print(f"  {'交易日':<24} {len(daily_rets):>16d} {'118':>16}")
print(f"  {'回测耗时':<24} {elapsed/60:>15.1f}分 {'—':>16}")
print(f"  {'—':<24} {'—':<16} {'—':<16}")
print(f"  {'累计收益':<24} {cum_ret*100:>15.2f}% {'29.17%':>16}")
print(f"  {'年化收益':<24} {ann_ret*100:>15.2f}% {'—':>16}")
print(f"  {'年化波动':<24} {ann_vol*100:>15.2f}% {'—':>16}")
print(f"  {'夏普比率':<24} {sharpe:>15.2f} {'1.20':>16}")
print(f"  {'最大回撤':<24} {max_dd*100:>15.2f}% {'20.00%':>16}")
print(f"  {'日胜率':<24} {win*100:>15.1f}% {'—':>16}")
print(f"  {'信息比率':<24} {ir:>15.2f} {'—':>16}")
print(f"  {'等权基准':<24} {bench_cum*100:>15.2f}% {'3.55%':>16}")
