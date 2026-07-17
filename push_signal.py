#!/usr/bin/env python3
"""
信号推送脚本 v2 — FTShare + akshare bond_zh_cov, 零依赖jisilu
============================================================
数据源:
  bond_zh_cov()         → 1034只转债列表 + 转股价/正股价/CB现价/正股代码 (1次API)
  bond_zh_hs_cov_spot() → 当日CB实时价 (备用校准, 1次API)
  st_fresh/parquet      → 正股历史252天日线 (波动率, FTShare一次性下载)
  ft_stock_candlesticks_batch → 当日正股收盘价 (每日增量, ~10次API)

时效性: 全部实时 — 转债列表/转股价/CB价/正股价 → 每日收盘后新鲜拉取
"""
import sys, os, json, pickle, time, hashlib, base64, re
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import requests, akshare as ak
import pyaes

from config import MCConfig
from pricer import CCBPricer, CCBParams
from push_to_cos import upload_signal, upload_log as upload_to_cos

# ===== 配置 =====
ST_CACHE_DIR = Path.home() / 'chenjunming' / 'quant' / 'cache' / 'st_fresh'
MCP_URL = "https://market.ft.tech/gateway/mcp"
PASSWORD  = "336018"

M = 5000
N_POS = 10

# ── 远程日志 ──
class LogCollector:
    """收集运行日志, 通过Gist对外暴露"""
    def __init__(self):
        self.entries = []
        self.start_time = datetime.now()
    
    def log(self, level: str, msg: str, **kwargs):
        entry = {
            'ts': datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'level': level,
            'msg': msg,
        }
        if kwargs:
            entry['detail'] = kwargs
        self.entries.append(entry)
        # 同时输出到stdout
        tag = {'ERROR':'❌','WARN':'⚠','OK':'✅','INFO':'•'}.get(level, '')
        print(f"    {tag} {msg}")
    
    def to_dict(self):
        return {
            'run_at': self.start_time.isoformat(),
            'duration_s': round((datetime.now()-self.start_time).total_seconds(),1),
            'total_entries': len(self.entries),
            'errors': [e for e in self.entries if e['level']=='ERROR'],
            'warnings': [e for e in self.entries if e['level']=='WARN'],
            'entries': self.entries,
        }

LOG = LogCollector()


def encrypt_aes(plaintext: str, password: str) -> str:
    key = hashlib.sha256(password.encode()).digest()
    iv = os.urandom(16)
    data = plaintext.encode('utf-8')
    pad_len = 16 - (len(data) % 16)
    padded = data + bytes([pad_len]) * pad_len
    cipher = pyaes.AESModeOfOperationCBC(key, iv=iv)
    ct = b''
    for i in range(0, len(padded), 16):
        ct += cipher.encrypt(padded[i:i+16])
    return base64.b64encode(iv + ct).decode('ascii')


def fetch_bond_metadata():
    """akshare bond_zh_cov — 一次调用: 转债列表+转股价+正股价+CB现价"""
    LOG.log('INFO', '拉取转债元数据 bond_zh_cov()')
    try:
        df = ak.bond_zh_cov()
        LOG.log('OK', f'bond_zh_cov: {len(df)} 只', total=len(df))
    except Exception as e:
        LOG.log('ERROR', f'bond_zh_cov拉取失败: {e}', source='akshare')
        return []
    
    bonds = []
    nan_count = 0
    for _, r in df.iterrows():
        try:
            conv = float(r['转股价'])
            st_p = float(r['正股价'])
            cb_p = float(r['债现价'])
            if np.isnan(conv) or np.isnan(st_p) or np.isnan(cb_p):
                nan_count += 1
                continue
            if conv <= 0 or st_p <= 0 or cb_p <= 0:
                nan_count += 1
                continue
            bonds.append({
                'cb_id': str(r['债券代码']).zfill(6),
                'name': str(r['债券简称']),
                'stock_id': str(r['正股代码']).zfill(6),
                'conv_price': conv,
                'stock_price': st_p,
                'cb_price': cb_p,
            })
        except (ValueError, KeyError):
            nan_count += 1
    
    LOG.log('OK', f'有效: {len(bonds)} 只', nan_dropped=nan_count)
    if nan_count > 0:
        LOG.log('WARN', f'{nan_count} 只因NaN/0/缺失被丢弃')
    return bonds


def fetch_live_cb_prices(bonds):
    """akshare bond_zh_hs_cov_spot — 当日实时CB价, 覆盖/校准bond_zh_cov数据"""
    print("  [2/3] 拉取CB实时价...")
    try:
        df = ak.bond_zh_hs_cov_spot()
        live = {}
        for _, r in df.iterrows():
            code = str(r.get('code', r.get('symbol', '')))
            price = float(r.get('trade', 0))
            if code and price > 0:
                live[code] = price
        # 覆盖
        updated = 0
        for b in bonds:
            cid = b['cb_id']
            if cid in live and live[cid] > 0:
                b['cb_price'] = live[cid]
                updated += 1
        print(f"    CB实时价覆盖: {updated}/{len(bonds)} 只 (spot={len(live)}只)")
    except Exception as e:
        print(f"    CB实时价拉取失败(保留bond_zh_cov价格): {e}")


def get_ftshare_session() -> str:
    """获取FTShare MCP session"""
    r = requests.post(MCP_URL, json={
        "jsonrpc":"2.0","id":1,"method":"initialize",
        "params":{"protocolVersion":"2025-11-25","capabilities":{},
                  "clientInfo":{"name":"cb-sig","version":"1"}}
    }, headers={"Accept":"application/json, text/event-stream",
                "Content-Type":"application/json"}, timeout=15)
    return r.headers.get('Mcp-Session-Id','').strip()


def ft_batch_stock_close(session_id: str, symbols: list[str], 
                         since_ts: int, until_ts: int) -> dict:
    """FTShare批量K线 → {ft_code: latest_close_price}"""
    r = requests.post(MCP_URL, json={
        "jsonrpc":"2.0","id":2,"method":"tools/call",
        "params":{"name":"ft_stock_candlesticks_batch","arguments":{
            "symbols":symbols,"interval_unit":"day",
            "since_ts_millis":since_ts,"until_ts_millis":until_ts,
            "adjust_kind":"forward"}}
    }, headers={"Accept":"application/json, text/event-stream",
                "Content-Type":"application/json",
                "Mcp-Session-Id":session_id}, timeout=30)
    
    match = re.search(r'data:\s*(\{.*\})', r.text, re.DOTALL)
    if not match: return {}
    d = json.loads(match.group(1))
    if d.get('isError'): return {}
    text = d['result']['content'][0]['text']
    
    # Parse markdown table → {symbol: latest_close}
    result = {}
    for line in text.split('\n'):
        if not line.startswith('| ') or '---' in line or 'close' in line.lower():
            continue
        p = [x.strip() for x in line.split('|') if x.strip()]
        if len(p) < 6: continue
        try:
            close = float(p[0])
            symbol = p[4]
            # Keep highest date's close
            if symbol not in result or close > 0:
                result[symbol] = close
        except (ValueError, IndexError):
            continue
    return result


def fetch_today_stock_close(bonds):
    """FTShare增量拉取当日正股收盘价, 更新parquet缓存"""
    print("  [3/3] 拉取正股当日收盘价 (FTShare)...")
    
    today = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
    since_ts = int((today - timedelta(days=2)).timestamp() * 1000)
    until_ts = int((today + timedelta(days=1)).timestamp() * 1000) - 1
    today_str = today.strftime('%Y-%m-%d')
    
    # 收集需要刷新的正股FTShare代码
    need_refresh = {}
    for b in bonds:
        sid = b['stock_id']
        code = sid.zfill(6)
        ft = f"{code}.SH" if code.startswith(('6','9')) else f"{code}.SZ"
        need_refresh[ft] = sid
    
    if not need_refresh:
        print("    无正股需要刷新")
        return
    
    sid = get_ftshare_session()
    if not sid:
        print("    FTShare session失败, 使用缓存价格")
        # Fallback to cache
        for b in bonds:
            fpath = ST_CACHE_DIR / f"{b['stock_id']}.parquet"
            if fpath.exists():
                df = pd.read_parquet(fpath)
                latest = df[df['date'] == df['date'].max()]
                if not latest.empty:
                    b['stock_price'] = float(latest['close'].iloc[-1])
        return
    
    # 批量拉取
    symbols = list(need_refresh.keys())
    BATCH = 20
    updates = {}
    
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i+BATCH]
        try:
            prices = ft_batch_stock_close(sid, batch, since_ts, until_ts)
            updates.update(prices)
        except Exception as e:
            print(f"    batch error: {e}")
        time.sleep(0.15)
    
    print(f"    FTShare返回: {len(updates)} 只当日收盘价")
    
    # 更新bonds + parquet缓存
    refreshed = 0
    for b in bonds:
        sid = b['stock_id']
        code = sid.zfill(6)
        ft_code = f"{code}.SH" if code.startswith(('6','9')) else f"{code}.SZ"
        
        if ft_code in updates and updates[ft_code] > 0:
            b['stock_price'] = updates[ft_code]
            # 增量写入parquet
            fpath = ST_CACHE_DIR / f"{sid}.parquet"
            new_row = pd.DataFrame([{'date': today_str, 'close': updates[ft_code]}])
            if fpath.exists():
                df = pd.read_parquet(fpath)
                if today_str not in df['date'].values:
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_parquet(fpath, index=False)
            else:
                new_row.to_parquet(fpath, index=False)
            refreshed += 1
    
    print(f"    已刷新: {refreshed}/{len(bonds)} 只")


def load_volatility_cache(bonds):
    """从parquet缓存计算每只正股的波动率"""
    vols = {}
    for b in bonds:
        sid = b['stock_id']
        fpath = ST_CACHE_DIR / f"{sid}.parquet"
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        closes = df['close'].values[-252:]
        if len(closes) < 20:
            continue
        lr = np.diff(np.log(np.maximum(closes, 0.01)))
        vol = float(np.std(lr) * np.sqrt(252))
        vols[sid] = max(0.05, min(vol, 1.5))
    return vols


def compute_signal():
    bonds = fetch_bond_metadata()
    fetch_live_cb_prices(bonds)
    fetch_today_stock_close(bonds)
    
    # 波动率缓存
    vols = load_volatility_cache(bonds)
    print(f"    波动率覆盖: {len(vols)} 只")
    
    # 过滤有效券
    valid = [b for b in bonds 
             if b['cb_price'] > 0 and b['stock_price'] > 0
             and b['conv_price'] > 0 and b['stock_id'] in vols]
    no_vol = [b for b in bonds if b['stock_id'] not in vols]
    if no_vol:
        print(f"    ⚠ 无波动率数据: {len(no_vol)} 只")
    print(f"    有效: {len(valid)} 只")
    
    if len(valid) < N_POS:
        print(f"    ❌ 有效券不足{N_POS}只, 退出")
        return None
    
    # MC定价
    print(f"  MC定价 (M={M}, GPU)...")
    config = MCConfig(n_paths=M, n_days=756, use_gpu=True)
    pricer = CCBPricer(config)
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    results = []
    for b in valid:
        try:
            ccb = CCBParams(
                name=b['name'], ticker=b['cb_id'], stock_ticker=b['stock_id'],
                face_value=100.0, conversion_price=b['conv_price'],
                days_to_maturity=252*2, call_trigger_pct=1.30,
                put_trigger_pct=0.70, down_trigger_pct=0.85,
                put_price=100.0, redemption_price=108.0,
                call_mc=15, call_nc=30, put_mp=30, put_np=30,
                market_price=b['cb_price'], stock_price=b['stock_price'],
                volatility=vols[b['stock_id']], conversion_start_day=0)
            mc_p, _ = pricer.price_single(ccb, n_paths=M, step_days=63, seed=42)
            results.append({
                'code': b['cb_id'], 'name': b['name'],
                'market_price': round(b['cb_price'], 2),
                'model_price': round(mc_p, 2),
                'discount': round((mc_p - b['cb_price']) / b['cb_price'] * 100, 2),
                'stock_price': round(b['stock_price'], 2),
                'conv_price': round(b['conv_price'], 2),
                'volatility': round(vols[b['stock_id']], 4),
            })
        except Exception as e:
            print(f"    {b['cb_id']}: {e}")
    
    results.sort(key=lambda x: x['discount'], reverse=True)
    return {
        'date': today_str,
        'time': datetime.now().strftime('%H:%M:%S'),
        'n_bonds_total': len(bonds),
        'n_bonds_priced': len(results),
        'n_positions': min(N_POS, len(results)),
        'positions': results[:N_POS],
        'buy': [r['code'] for r in results[:N_POS]],
    }


if __name__ == '__main__':
    LOG.log('INFO', '可转债MC定价信号推送 启动')
    t0 = time.time()
    
    try:
        signal = compute_signal()
    except Exception as e:
        LOG.log('ERROR', f'compute_signal崩溃: {e}', trace=str(e))
        signal = None
    
    if signal:
        buy_list = signal['buy']
        LOG.log('OK', f'Top{N_POS}: {buy_list}', n_priced=signal['n_bonds_priced'])
        
        print(f"\n  今日Top{N_POS}:")
        for i, p in enumerate(signal['positions']):
            print(f"  {i+1}. {p['code']} {p['name'][:10]:10s} "
                  f"MC={p['model_price']:.1f} MKT={p['market_price']:.1f} "
                  f"disc={p['discount']:+.1f}%")
        
        encrypted = encrypt_aes(json.dumps(signal, ensure_ascii=False), PASSWORD)
        try:
            upload_signal({"encrypted": encrypted})
            LOG.log('OK', '信号已上传COS')
        except Exception as e:
            LOG.log('ERROR', f'COS上传信号失败: {e}')
    else:
        LOG.log('ERROR', '无有效信号, 跳过上传')
    
    # 上传运行日志
    log_data = LOG.to_dict()
    log_data['runtime_sec'] = round(time.time() - t0, 1)
    try:
        upload_to_cos(log_data)
        d, n = log_data['duration_s'], log_data['total_entries']
        LOG.log('INFO', f'日志已上传COS ({d}s, {n}条)')
    except Exception as e:
        LOG.log('ERROR', f'COS上传日志失败: {e}')
