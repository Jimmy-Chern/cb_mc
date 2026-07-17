#!/usr/bin/env python3
"""补全全部309只转债历史数据到缓存"""
import pickle, time, sys
from pathlib import Path
import pandas as pd
import akshare as ak
from tqdm import tqdm

CACHE_DIR = Path.home() / 'chenjunming/quant/cache'
JISILU = Path.home() / 'chenjunming/quant/jisilu.txt'

# Load existing cache
cb_hist = pickle.load(open(CACHE_DIR/'cb_hist.pkl','rb')) if (CACHE_DIR/'cb_hist.pkl').exists() else {}
st_hist = pickle.load(open(CACHE_DIR/'st_hist.pkl','rb')) if (CACHE_DIR/'st_hist.pkl').exists() else {}

# Parse all jisilu bonds
lines = [l.strip() for l in open(JISILU) if '\t' in l and l.split('\t')[0].isdigit()]
all_bonds = []
for l in lines:
    f = l.split('\t')
    if len(f) >= 10:
        all_bonds.append({'cb_id': f[0], 'stock_id': f[4], 'name': f[1]})

print(f'jisilu: {len(all_bonds)} 只, 已有CB: {len(cb_hist)}, 已有Stock: {len(st_hist)}')

# Download missing
new_cb = 0; new_st = 0
for b in tqdm(all_bonds, desc='补全数据'):
    cb_id, st_id = b['cb_id'], b['stock_id']
    cb_pfx = 'sh' if cb_id.startswith('11') else 'sz'
    st_pfx = 'sh' if st_id.startswith(('6','5')) else 'sz'
    
    if cb_id not in cb_hist:
        try:
            df = ak.bond_zh_hs_cov_daily(symbol=f'{cb_pfx}{cb_id}')
            if len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                cb_hist[cb_id] = df.set_index('date')
                new_cb += 1
        except: pass
        time.sleep(0.03)
    
    if cb_id not in st_hist:
        try:
            df = ak.stock_zh_a_daily(symbol=f'{st_pfx}{st_id}', adjust='qfq')
            if len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                st_hist[cb_id] = df.set_index('date')
                new_st += 1
        except: pass
        time.sleep(0.03)

# Save
pickle.dump(cb_hist, open(CACHE_DIR/'cb_hist.pkl','wb'))
pickle.dump(st_hist, open(CACHE_DIR/'st_hist.pkl','wb'))
print(f'新增CB: {new_cb}, 新增Stock: {new_st}')
print(f'总计CB: {len(cb_hist)}, Stock: {len(st_hist)}')

# Count coverage for 2023 period
target_start = pd.Timestamp('2023-02-01'); target_end = pd.Timestamp('2023-08-01')
count = 0
for cid in sorted(set(cb_hist.keys()) & set(st_hist.keys())):
    if (cb_hist[cid].index.min() <= target_start and cb_hist[cid].index.max() >= target_end and
        st_hist[cid].index.min() <= target_start and st_hist[cid].index.max() >= target_end):
        count += 1
print(f'2023年2-7月覆盖: {count} 只')
