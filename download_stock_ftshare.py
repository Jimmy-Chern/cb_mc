#!/usr/bin/env python3
"""
正股历史日线下载 — FTShare MCP ft_stock_candlesticks_batch
============================================================
API限制: since/until ≤3天 → 252天分84段
每段: 20只股票×3天, 批间sleep 0.3s
总量: (309/20)×84 ≈ 1300次调用 ≈ 6-7分钟
缓存: ~/chenjunming/quant/cache/st_fresh/{code}.parquet
"""
import requests, json, time, re, pickle
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

MCP_URL = "https://market.ft.tech/gateway/mcp"
CACHE_DIR = Path.home() / 'chenjunming' / 'quant' / 'cache' / 'st_fresh'

BATCH_SIZE = 20
DAYS_PER_CALL = 3
HISTORY_DAYS = 252
SLEEP_S = 0.3


def get_session() -> str:
    r = requests.post(MCP_URL, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                   "clientInfo": {"name": "cb-dl", "version": "1.0"}}
    }, headers={"Accept": "application/json, text/event-stream",
                "Content-Type": "application/json"}, timeout=15)
    sid = r.headers.get('Mcp-Session-Id', '')
    if not sid: raise Exception(f"init failed: {r.status_code}")
    return sid.strip()


def call_batch_candles(session_id: str, symbols: list[str], since_ts: int, until_ts: int) -> str:
    """返回 markdown 表格文本"""
    r = requests.post(MCP_URL, json={
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "ft_stock_candlesticks_batch", "arguments": {
            "symbols": symbols,
            "interval_unit": "day",
            "since_ts_millis": since_ts,
            "until_ts_millis": until_ts,
            "adjust_kind": "forward",
        }}
    }, headers={
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "Mcp-Session-Id": session_id,
    }, timeout=30)
    
    # Parse SSE: extract JSON from data: {...} line
    match = re.search(r'data:\s*(\{.*\})', r.text, re.DOTALL)
    if not match:
        return ""
    data = json.loads(match.group(1))
    if data.get('isError'):
        return ""
    return data['result']['content'][0]['text']


def parse_table(text: str) -> list[dict]:
    """解析 markdown 表格 -> [{'symbol':'601166.SH','date':'2026-07-13','open':...,...}]"""
    rows = []
    lines = text.split('\n')
    # 跳过标题行和分隔行
    for line in lines:
        if not line.startswith('| ') or '---' in line or 'close' in line.lower():
            continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) < 9:
            continue
        try:
            close = float(parts[0])
            high = float(parts[1])
            low = float(parts[2])
            open_p = float(parts[3])
            symbol = parts[4]
            ts_ms = int(parts[5])
            volume = int(float(parts[8])) if len(parts) > 8 else 0
            
            # timestamp -> date string
            dt = datetime.fromtimestamp(ts_ms / 1000)
            rows.append({
                'symbol': symbol,
                'date': dt.strftime('%Y-%m-%d'),
                'open': open_p, 'high': high, 'low': low, 'close': close,
                'volume': volume,
            })
        except (ValueError, IndexError):
            continue
    return rows


def stock_code_to_ftshare(code: str) -> str:
    code = code.zfill(6)
    if code.startswith(('6', '9')):
        return f"{code}.SH"
    return f"{code}.SZ"

def get_all_stock_codes() -> list[str]:
    """从 bond_zh_cov() 获取所有正股代码 (替代jisilu)"""
    import akshare as ak
    df = ak.bond_zh_cov()
    codes = set()
    for _, r in df.iterrows():
        try:
            sid = str(r['正股代码']).zfill(6)
            if sid.isdigit() and len(sid) == 6:
                codes.add(sid)
        except: continue
    return sorted(codes)


def load_df(code: str) -> pd.DataFrame:
    f = CACHE_DIR / f"{code}.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame()


def save_df(code: str, df: pd.DataFrame):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_DIR / f"{code}.parquet", index=False)


def main():
    stock_codes = get_all_stock_codes()
    
    # 增量: 跳过已有数据的正股
    new_codes = [c for c in stock_codes if not (CACHE_DIR / f"{c}.parquet").exists()]
    if new_codes:
        print(f"正股: {len(stock_codes)} 只 (新增 {len(new_codes)} 只, 已缓存 {len(stock_codes)-len(new_codes)} 只)")
        stock_codes = new_codes
    else:
        print(f"正股: {len(stock_codes)} 只 (全部已缓存)")
    
    if not stock_codes:
        print("无需下载!")
        return
    ft_codes = {c: stock_code_to_ftshare(c) for c in stock_codes}
    
    # 日期分段
    end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    chunks = []
    c = end - timedelta(days=HISTORY_DAYS)
    while c < end:
        ce = min(c + timedelta(days=DAYS_PER_CALL - 1), end - timedelta(days=1))
        chunks.append((c, ce))
        c = ce + timedelta(days=1)
    print(f"日期分段: {len(chunks)} (每段≤{DAYS_PER_CALL}天)")
    print(f"时间范围: {chunks[0][0].date()} ~ {chunks[-1][1].date()}")
    
    sid = get_session()
    
    manifest = {}
    codes = list(ft_codes.keys())
    total_batches = len(chunks) * ((len(codes) + BATCH_SIZE - 1) // BATCH_SIZE)
    
    pbar = tqdm(total=total_batches, desc="下载正股日线")
    pbar.set_postfix({'done': '0'})
    
    for d_start, d_end in chunks:
        since_ts = int(d_start.timestamp() * 1000)
        until_ts = int((d_end + timedelta(days=1)).timestamp() * 1000) - 1
        
        for bi in range(0, len(codes), BATCH_SIZE):
            batch_codes = codes[bi:bi+BATCH_SIZE]
            batch_syms = [ft_codes[c] for c in batch_codes]
            
            try:
                text = call_batch_candles(sid, batch_syms, since_ts, until_ts)
                rows = parse_table(text)
                
                # 按code分组
                by_code = {}
                for r in rows:
                    code = r['symbol'].replace('.SH', '').replace('.SZ', '')
                    by_code.setdefault(code, []).append(r)
                
                for code, candles in by_code.items():
                    new_df = pd.DataFrame(candles).drop(columns=['symbol'])
                    old_df = load_df(code)
                    merged = (pd.concat([old_df, new_df]) if not old_df.empty 
                             else new_df).drop_duplicates('date').sort_values('date')
                    save_df(code, merged)
                    manifest[code] = len(merged)
                
                pbar.update(1)
                pbar.set_postfix({'codes': len(manifest), 'last': batch_codes[0]})
                time.sleep(SLEEP_S)
                
            except Exception as e:
                pbar.set_postfix({'err': str(e)[:30]})
                time.sleep(2)
    
    pbar.close()
    pickle.dump(manifest, open(CACHE_DIR / 'manifest.pkl', 'wb'))
    print(f"\n完成: {len(manifest)} 只正股")
    print(f"缓存: {CACHE_DIR}")


if __name__ == '__main__':
    main()
