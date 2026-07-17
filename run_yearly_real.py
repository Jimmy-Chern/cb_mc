#!/usr/bin/env python3
"""A股可转债 一年回测 — 真实参数 + GBM合成行情"""
import sys, os, json, time
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from tqdm import tqdm
from config import MCConfig
from pricer import CCBPricer, CCBParams

print("="*60)
print("A股可转债 MC定价 一年回测 (真实参数)")
print("="*60)

# ── 30只真实转债参数(from MCP) ──
bonds_real = [
    # (cb_id, name, stock_price, conv_price, cb_price, volatility, call_pct, put_pct)
    ("113052","兴业转债",17.33,20.13,115.58,0.223,1.30,0.70),
    ("113051","节能转债",3.44,3.33,127.51,0.360,1.30,0.70),
    ("113046","金田转债",10.70,10.18,133.62,0.310,1.30,0.70),
    ("113043","财通转债",8.53,7.98,130.91,0.280,1.30,0.70),
    ("113048","晶科转债",3.96,4.47,122.58,0.422,1.30,0.70),
    ("113056","重银转债",9.79,9.22,125.10,0.298,1.30,0.70),
    ("113042","上银转债",8.91,8.35,115.58,0.210,1.30,0.70),
    ("113049","长汽转债",15.26,39.18,107.61,0.364,1.30,0.70),
    ("113054","绿动转债",6.94,8.67,123.02,0.393,1.30,0.70),
    ("113058","友发转债",5.01,4.48,132.00,0.330,1.30,0.70),
    ("113059","福莱转债",9.73,41.56,119.28,0.335,1.30,0.70),
    ("111015","东亚转债",16.39,20.28,125.77,0.350,1.30,0.70),
    ("111003","聚合转债",8.49,11.31,131.61,0.450,1.30,0.70),
    ("111004","明新转债",16.48,24.37,125.49,0.380,1.30,0.70),
    ("111009","盛泰转债",5.86,10.51,122.51,0.420,1.30,0.70),
    ("111012","福新转债",29.91,11.07,324.21,0.520,1.30,0.70),
    ("111014","李子转债",8.54,18.14,120.59,0.400,1.30,0.70),
    ("111017","蓝天转债",6.44,7.98,126.42,0.350,1.30,0.70),
    ("111018","华康转债",10.54,15.96,124.09,0.370,1.30,0.70),
    ("111020","合顺转债",8.49,10.53,133.34,0.613,1.30,0.70),
    ("111021","奥锐转债",20.21,24.60,136.59,0.400,1.30,0.70),
    ("111022","锡振转债",24.69,22.39,158.25,0.450,1.30,0.70),
    ("111024","澳弘转债",28.66,33.54,160.81,0.624,1.30,0.70),
    ("111000","起帆转债",20.90,17.29,156.21,0.480,1.30,0.70),
    ("111002","特纸转债",15.00,13.49,142.87,0.440,1.30,0.70),
    ("113053","隆22转债",12.11,17.50,121.99,0.380,1.30,0.70),
    ("113037","紫银转债",2.49,3.55,109.90,0.250,1.30,0.70),
    ("113039","嘉泽转债",4.21,2.89,155.75,0.360,1.30,0.70),
    ("110099","福能转债",10.34,9.84,141.66,0.330,1.30,0.70),
    ("110100","龙建转债",3.13,4.63,122.56,0.350,1.30,0.70),
]

N_DAYS = 252  # 1 year
N_PATHS = 50  # MC paths for pricing

# ── Generate synthetic 1-year price paths ──
print(f"\n生成 {len(bonds_real)} 只转债 × {N_DAYS} 天 合成行情...")
np.random.seed(42)

# Stock paths: GBM with real vol
stock_paths = {}
cb_paths = {}
for cb_id, name, sp, cp, cbp, vol, call_p, put_p in bonds_real:
    daily_sigma = vol / np.sqrt(252)
    daily_drift = 0.03 / 252  # 3% annual drift
    
    # Stock GBM
    log_rets_st = daily_drift - 0.5*daily_sigma**2 + daily_sigma * np.random.randn(N_DAYS)
    stock_paths[cb_id] = sp * np.exp(np.cumsum(log_rets_st))
    
    # CB price: simplified model = max(bond_floor, conversion_value) * (1 + noise)
    bond_floor = 95 + 5*np.random.random()  # Random bond floor 95-100
    conv_ratio = 100.0 / cp
    cv = conv_ratio * stock_paths[cb_id]
    
    # CB premium: mean-reverting around observed premium
    obs_premium = cbp / max(cv[0], 1)  # observed premium at start
    premiums = np.ones(N_DAYS) * obs_premium
    for i in range(1, N_DAYS):
        premiums[i] = premiums[i-1] + 0.05*(obs_premium - premiums[i-1]) + 0.01*np.random.randn()
        premiums[i] = max(0.8, premiums[i])  # Floor
    
    cb_paths[cb_id] = np.maximum(bond_floor, cv) * premiums

print("  完成")

# ── Backtest ──
print(f"\n回测 {N_DAYS} 天...")
config = MCConfig(n_paths=N_PATHS, n_days=252, use_gpu=True)
pricer = CCBPricer(config)

portfolio = [1.0]
daily_rets = []
prev_holdings = set()
tc = 0.001

for day in tqdm(range(0, N_DAYS-1), desc="回测"):
    # Price all bonds
    discounts = []
    for cb_id, name, sp_init, cp, cbp_init, vol, call_p, put_p in bonds_real:
        stock_p = stock_paths[cb_id][day]
        cb_p = cb_paths[cb_id][day]
        if stock_p <= 0 or cb_p <= 0: continue
        
        # Forward-looking vol (use next 20 days)
        lookback = stock_paths[cb_id][max(0,day-20):day+1]
        if len(lookback) >= 5:
            lr = np.diff(np.log(lookback))
            v = float(np.std(lr)*np.sqrt(252))
            v = max(0.05, min(v, 1.5))
        else: v = vol
        
        ccb = CCBParams(name=name, ticker=cb_id, stock_ticker="",
                       face_value=100.0, conversion_price=cp,
                       days_to_maturity=max(252, N_DAYS-day),
                       call_trigger_pct=call_p, put_trigger_pct=put_p,
                       down_trigger_pct=0.85, put_price=100.0, redemption_price=108.0,
                       market_price=cb_p, stock_price=stock_p,
                       volatility=v, conversion_start_day=0, industry="通用")
        try:
            mc_p, _ = pricer.price_single(ccb, n_paths=N_PATHS, step_days=63, seed=42)
            disc = (mc_p - cb_p) / cb_p if cb_p > 0 else -1
            discounts.append({"cb_id": cb_id, "disc": disc, "cb_p": cb_p, "cb_path": cb_paths[cb_id], "day": day})
        except: continue
    
    if len(discounts) < 10: continue
    
    # Top 10
    discounts.sort(key=lambda x: x["disc"], reverse=True)
    top10 = discounts[:10]
    top10_ids = set(d["cb_id"] for d in top10)
    
    # Next-day returns
    pos_rets = []
    for d in top10:
        tmr_p = d["cb_path"][day+1]
        if tmr_p > 0:
            pos_rets.append((tmr_p - d["cb_p"]) / d["cb_p"])
    
    if pos_rets:
        port_ret = np.mean(pos_rets)
        port_ret -= tc * len(top10_ids - prev_holdings) / 10
        prev_holdings = top10_ids
    else:
        port_ret = 0
    
    daily_rets.append(port_ret)
    portfolio.append(portfolio[-1] * (1 + port_ret))

# ── Benchmark: equal-weight ──
bench_rets = []
for day in range(N_DAYS-1):
    day_rets = []
    for cb_id, *_ in bonds_real:
        p_t = cb_paths[cb_id][day]; p_tm = cb_paths[cb_id][day+1]
        if p_t > 0: day_rets.append((p_tm-p_t)/p_t)
    bench_rets.append(np.mean(day_rets) if day_rets else 0)

bench_cum = (1+pd.Series(bench_rets)).prod() - 1

# ── Results ──
ret_s = pd.Series(daily_rets)
cum_ret = portfolio[-1] - 1
ann_ret = (1+cum_ret)**(252/len(daily_rets))-1
ann_vol = ret_s.std()*np.sqrt(252)
sharpe = ann_ret/ann_vol if ann_vol>0 else 0
cum_s = (1+ret_s).cumprod()
max_dd = (cum_s/cum_s.expanding().max()-1).min()
win = (ret_s>0).mean()
info_ratio = (ret_s.mean() - np.mean(bench_rets)) / ret_s.std() * np.sqrt(252) if ret_s.std()>0 else 0

print("\n" + "="*60)
print("  A股可转债 MC定价 一年回测 结果")
print("="*60)
print(f"  {'指标':<22} {'LSM策略':<15} {'等权基准':<15}")
print(f"  {'累计收益':<22} {cum_ret*100:>14.2f}% {bench_cum*100:>14.2f}%")
print(f"  {'年化收益':<22} {ann_ret*100:>14.2f}% {'—':>15}")
print(f"  {'年化波动':<22} {ann_vol*100:>14.2f}% {'—':>15}")
print(f"  {'夏普比率':<22} {sharpe:>14.2f} {'—':>15}")
print(f"  {'最大回撤':<22} {max_dd*100:>14.2f}% {'—':>15}")
print(f"  {'日胜率':<22} {win*100:>14.1f}% {'—':>15}")
print(f"  {'信息比率':<22} {info_ratio:>14.2f} {'—':>15}")
print(f"\n  转债: {len(bonds_real)}只, 交易日: {len(daily_rets)}天, MC路径: {N_PATHS}条")

