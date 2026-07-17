#!/usr/bin/env python3
"""
信号服务器 — 对外提供HTTP接口，返回今日MC定价Top10
启动: .venv/bin/python server_signal.py --port 8080
"""
import sys, os, json, pickle, time
from pathlib import Path
from flask import Flask, jsonify

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from config import MCConfig
from pricer import CCBPricer, CCBParams

app = Flask(__name__)

CACHE_DIR = Path.home() / 'chenjunming/quant/cache'
JISILU    = Path.home() / 'chenjunming/quant/jisilu.txt'
SIGNAL_FILE = Path(__file__).parent / 'latest_signal.json'

M = 5000       # 5000 paths matching paper
N_POS = 10     # Top 10 holdings


def load_bonds():
    """Load bond parameters from jisilu + cache."""
    cb_hist = pickle.load(open(CACHE_DIR/'cb_hist.pkl','rb'))
    st_hist = pickle.load(open(CACHE_DIR/'st_hist.pkl','rb'))
    
    lines = [l.strip() for l in open(JISILU) if '\t' in l and l.split('\t')[0].isdigit()]
    bonds = []
    for l in lines:
        f = l.split('\t')
        if len(f) < 10: continue
        cb_id = f[0]; stock_id = f[4]
        try:
            conv_price = float(f[9].replace(',',''))
            cb_price   = float(f[2].replace(',',''))
            stock_price = float(f[6].replace(',',''))
            name = f[1]
            if conv_price > 0 and stock_price > 0 and cb_price > 0:
                bonds.append({
                    'cb_id': cb_id, 'stock_id': stock_id, 'name': name,
                    'conv_price': conv_price, 'cb_price': cb_price,
                    'stock_price': stock_price
                })
        except: continue
    return bonds, cb_hist, st_hist


def compute_signal():
    """Run MC pricing on all bonds, return Top N."""
    print(f"[{time.strftime('%H:%M:%S')}] 开始MC定价...")
    bonds, cb_hist, st_hist = load_bonds()
    
    valid = [b for b in bonds if b['cb_id'] in cb_hist and b['cb_id'] in st_hist]
    print(f"  有效转债: {len(valid)} 只")
    
    config = MCConfig(n_paths=M, n_days=756, use_gpu=True)
    pricer = CCBPricer(config)
    
    results = []
    for b in valid:
        cid = b['cb_id']
        # Get latest stock price and vol
        st_df = st_hist[cid]
        if len(st_df) < 20: continue
        closes = st_df['close'].values[-252:]
        stock_p = closes[-1]
        if len(closes) >= 20:
            lr = np.diff(np.log(closes))
            vol = float(np.std(lr) * np.sqrt(252))
            vol = max(0.05, min(vol, 1.5))
        else: vol = 0.3
        
        # Get latest CB price
        cb_df = cb_hist[cid]
        cb_p = float(cb_df['close'].iloc[-1])
        
        ccb = CCBParams(name=b['name'], ticker=cid, stock_ticker=b['stock_id'],
                       face_value=100.0, conversion_price=b['conv_price'],
                       days_to_maturity=252*2, call_trigger_pct=1.30,
                       put_trigger_pct=0.70, down_trigger_pct=0.85,
                       put_price=100.0, redemption_price=108.0,
                       call_mc=15, call_nc=30, put_mp=30, put_np=30,
                       market_price=cb_p, stock_price=stock_p,
                       volatility=vol, conversion_start_day=0)
        try:
            mc_p, _ = pricer.price_single(ccb, n_paths=M, step_days=63, seed=42)
            disc = (mc_p - cb_p) / cb_p
            results.append({
                'code': cid, 'name': b['name'],
                'market_price': round(cb_p, 2),
                'model_price': round(mc_p, 2),
                'discount': round(disc * 100, 2),
                'stock_price': round(stock_p, 2),
                'conv_price': round(b['conv_price'], 2),
                'volatility': round(vol, 4),
            })
        except Exception as e:
            print(f"  {cid} 定价失败: {e}")
    
    results.sort(key=lambda x: x['discount'], reverse=True)
    top10 = results[:N_POS]
    
    print(f"  定价完成: {len(results)} 只, Top10:")
    for i, r in enumerate(top10):
        print(f"    {i+1}. {r['code']} {r['name'][:10]:10s} "
              f"MC={r['model_price']:.1f} MKT={r['market_price']:.1f} "
              f"disc={r['discount']:+.1f}%")
    
    return top10, results


# ── API Endpoints ──

@app.route('/signal')
def get_signal():
    """返回今日Top10买入信号"""
    signal, _ = compute_signal()
    return jsonify({
        'date': time.strftime('%Y-%m-%d'),
        'time': time.strftime('%H:%M:%S'),
        'n_positions': len(signal),
        'positions': signal,
        'buy': [s['code'] for s in signal],
    })


@app.route('/signal/all')
def get_all():
    """返回全部转债的定价结果"""
    _, all_results = compute_signal()
    return jsonify({
        'date': time.strftime('%Y-%m-%d'),
        'total': len(all_results),
        'results': all_results,
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'time': time.strftime('%Y-%m-%d %H:%M:%S')})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
    
    print(f"信号服务器启动: http://{args.host}:{args.port}")
    print(f"  GET /signal     → Top10买入信号")
    print(f"  GET /signal/all → 全部转债定价")
    print(f"  GET /health     → 健康检查")
    
    app.run(host=args.host, port=args.port, debug=False)
