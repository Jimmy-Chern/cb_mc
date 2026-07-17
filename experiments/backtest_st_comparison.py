#!/usr/bin/env python3
"""
回测对比: 全部Top10 vs 剔除ST Top10
====================================
数据: ~8个月 (2025-11 ~ 2026-07), ~160交易日
简化: 每月定价一次, 选出Top10, 持有到下个月
"""
import sys, pickle, os
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MCConfig
from pricer import CCBPricer, CCBParams
import akshare as ak

ST_CACHE = Path.home() / 'chenjunming' / 'quant' / 'cache' / 'st_fresh'
M = 5000; N_POS = 10; STEP = 63; REBAL = 'monthly'  # 月频调仓

def load_bonds_and_prices():
    """加载转债+正股历史价格"""
    print("[1/4] 加载数据...")
    df_bonds = ak.bond_zh_cov()
    
    # Load old caches (matching date ranges, 2020-2026)
    old_cb = pickle.load(open(Path.home()/'chenjunming/quant/cache/cb_hist.pkl','rb'))
    old_st = pickle.load(open(Path.home()/'chenjunming/quant/cache/st_hist.pkl','rb'))
    
    # Parse bonds with ST flags
    bonds = []
    for _, r in df_bonds.iterrows():
        try:
            cid = str(r['债券代码']).zfill(6)
            sid = str(r['正股代码']).zfill(6)
            sname = str(r['正股简称'])
            is_st = 'ST' in sname or '*ST' in sname
            bonds.append({
                'cb_id': cid, 'name': str(r['债券简称']), 'stock_id': sid,
                'conv_price': float(r['转股价']), 'is_st': is_st
            })
        except: continue
    
    # Build price matrix — both from old pickle caches for matching date ranges
    cb_prices = {}
    st_prices = {}
    valid = []
    for b in bonds:
        cid, sid = b['cb_id'], b['stock_id']
        if cid not in old_cb or cid not in old_st: continue  # old_st keyed by cb_id
        df_cb = old_cb[cid]
        df_st = old_st[cid]  # keyed by cb_id
        if len(df_cb) < 20 or len(df_st) < 20: continue
        df_cb.index = df_cb.index.astype(str).str[:10]
        df_st.index = df_st.index.astype(str).str[:10]
        cb_prices[cid] = df_cb['close']
        st_prices[sid] = df_st['close']
        valid.append(b)
    
    print(f"  债券: {len(valid)} 只, ST: {sum(1 for b in valid if b['is_st'])} 只")
    return valid, cb_prices, st_prices

def build_date_index(cb_prices):
    """找所有债券共同覆盖的交易日"""
    from collections import Counter
    dc = Counter()
    for cid, s in cb_prices.items():
        for d in s.index: dc[d] += 1
    # 至少100只债券有数据的日期
    dates = sorted(d for d, c in dc.items() if c >= 100)
    print(f"  交易日: {len(dates)} 天 ({dates[0]} ~ {dates[-1]})")
    return dates

def calc_volatility(st_prices, sid, ref_date):
    """计算截至ref_date的252天波动率"""
    s = st_prices.get(sid)
    if s is None: return None
    rd_str = str(ref_date)[:10]
    mask = s.index.astype(str).str[:10] <= rd_str
    if mask.sum() < 20: return None
    closes = s[mask].values[-252:]
    lr = np.diff(np.log(np.maximum(closes, 0.01)))
    vol = float(np.std(lr) * np.sqrt(252))
    return max(0.05, min(vol, 1.5))

def price_bonds_at_date(bonds, cb_prices, st_prices, date, pricer):
    """在指定日期MC定价所有转债"""
    date_str = str(date)[:10]  # Always use string
    results = []
    for b in bonds:
        cid, sid = b['cb_id'], b['stock_id']
        cb_s = cb_prices[cid]
        if date_str not in cb_s.index:
            continue
        cb_p = float(cb_s[date_str])
        if cb_p <= 50: continue
        
        # Stock price from parquet cache (date_str matching)
        st_s = st_prices.get(sid)
        if st_s is None: continue
        if date_str not in st_s.index: continue
        st_p = float(st_s[date_str])
        if st_p <= 0: continue
        
        vol = calc_volatility(st_prices, sid, date)
        if vol is None: continue
        
        try:
            ccb = CCBParams(
                name=b['name'], ticker=cid, stock_ticker=sid,
                face_value=100.0, conversion_price=b['conv_price'],
                days_to_maturity=252*2, call_trigger_pct=1.30,
                put_trigger_pct=0.70, down_trigger_pct=0.85,
                put_price=100.0, redemption_price=108.0,
                call_mc=15, call_nc=30, put_mp=30, put_np=30,
                market_price=cb_p, stock_price=st_p,
                volatility=vol, conversion_start_day=0)
            mc_p, _ = pricer.price_single(ccb, n_paths=M, step_days=STEP, seed=42)
            disc = (mc_p - cb_p) / cb_p * 100
            results.append({
                'cb_id': cid, 'is_st': b['is_st'],
                'market_price': cb_p, 'model_price': mc_p, 'discount': disc
            })
        except: continue
    return results

def run_backtest(label, bonds, cb_prices, st_prices, dates, strategy_filter):
    """回测: 每月初定价选Top10, 持有整月"""
    config = MCConfig(n_paths=M, n_days=756, use_gpu=True)
    pricer = CCBPricer(config)
    
    # 每月第一天
    rebal_dates = []
    last_month = None
    for d in dates:
        m = str(d)[:7]  # YYYY-MM from Timestamp
        if m != last_month:
            rebal_dates.append(d)
            last_month = m
    
    print(f"\n  [{label}] 调仓月: {len(rebal_dates)}")
    
    portfolio = [1.0]
    current_holdings = []
    
    for i, rd in enumerate(rebal_dates):
        # 定价
        eligible = [b for b in bonds if strategy_filter(b)]
        results = price_bonds_at_date(eligible, cb_prices, st_prices, rd, pricer)
        results.sort(key=lambda x: x['discount'], reverse=True)
        top10 = results[:N_POS]
        current_holdings = [r['cb_id'] for r in top10]
        
        if i == 0:
            print(f"    {rd}: 定价{len(results)}只, Top10={current_holdings[:3]}...")
            st_in = sum(1 for r in top10 if r['is_st'])
            discs = [r['discount'] for r in top10]
            print(f"    ST:{st_in}/10, 折价:{np.mean(discs):.1f}%")
        
        if i < len(rebal_dates) - 1:
            end_date = rebal_dates[i+1]
        else:
            end_date = dates[-1]
        
        # 持有到下次调仓
        period_dates = [str(d)[:10] for d in dates if str(rd)[:10] <= str(d)[:10] < str(end_date)[:10]]
        
        # 计算组合日收益
        daily_rets = []
        for j, d in enumerate(period_dates):
            if j == 0: continue  # skip first day (entry day)
            prev_d = period_dates[j-1]
            day_rets = []
            for cid in current_holdings:
                if cid not in cb_prices: continue
                cb_s = cb_prices[cid]
                if prev_d not in cb_s.index or d not in cb_s.index: continue
                prev, curr = float(cb_s[prev_d]), float(cb_s[d])
                if prev > 0:
                    day_rets.append((curr - prev) / prev)
            if day_rets:
                daily_rets.append(np.mean(day_rets))
        
        if i == 0:
            print(f"    period: {len(period_dates)}天, daily_rets: {len(daily_rets)}条")
        
        for r in daily_rets:
            portfolio.append(portfolio[-1] * (1 + r))
    
    total_ret = (portfolio[-1] - 1) * 100
    rets = pd.Series(np.diff(np.log(np.array(portfolio))))
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    
    print(f"    累计收益: {total_ret:+.2f}%, 夏普: {sharpe:.2f}")
    return total_ret, sharpe, portfolio

if __name__ == '__main__':
    valid, cb_prices, st_prices = load_bonds_and_prices()
    dates = build_date_index(cb_prices)
    
    if len(dates) < 60:
        print("交易日不足60天, 无法回测"); sys.exit(1)
    
    ret_a, sharpe_a, pf_a = run_backtest(
        "策略A: 全部转债", valid, cb_prices, st_prices, dates,
        lambda b: True)
    
    ret_b, sharpe_b, pf_b = run_backtest(
        "策略B: 剔除ST", valid, cb_prices, st_prices, dates,
        lambda b: not b['is_st'])
    
    print(f"\n{'='*60}")
    print(f"  结果对比")
    print(f"{'='*60}")
    print(f"  策略A (含ST):  累计 {ret_a:+.2f}%, 夏普 {sharpe_a:.2f}")
    print(f"  策略B (去ST):  累计 {ret_b:+.2f}%, 夏普 {sharpe_b:.2f}")
    print(f"  差异:          {ret_a - ret_b:+.2f}%")
    print(f"  结论: {'含ST更优' if ret_a > ret_b else '去ST更优'}")
