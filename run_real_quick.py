#!/usr/bin/env python3
"""FTShare MCP real-data CCB pricing — quick end-to-end."""
import sys, os, json, time
from pathlib import Path
import httpx, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from config import MCConfig
from pricer import CCBPricer, CCBParams

EP = "https://market.ft.tech/gateway/mcp"
H = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def parse_md(text):
    rows = []; hdr = None
    for l in text.split("\n"):
        if not l.startswith("|"): continue
        cols = [c.strip() for c in l.split("|")[1:-1]]
        if "---" in l: continue
        if hdr is None: hdr = cols; continue
        if hdr: rows.append(dict(zip(hdr, cols)))
    return rows


def main():
    client = httpx.Client(timeout=30)
    r = client.post(EP, json={"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"cb","version":"1"}},"id":1}, headers=H)
    sid = r.headers.get("mcp-session-id","")
    client.post(EP, json={"jsonrpc":"2.0","method":"notifications/initialized","id":2}, headers={**H,"mcp-session-id":sid})
    print(f"MCP session: {sid[:12]}...")

    def call(tool, args=None):
        r = client.post(EP, json={"jsonrpc":"2.0","method":"tools/call","params":{"name":tool,"arguments":args or {}},"id":int(time.time()*1000)%100000}, headers={**H,"mcp-session-id":sid})
        for line in r.text.split("\n"):
            if line.startswith("data:") and '"text"' in line:
                try:
                    d = json.loads(line[5:].strip())
                    for c in d.get("result",{}).get("content",[]):
                        if c.get("type")=="text": return c["text"]
                except: pass
        return ""

    raw = call("ft_get_cb_lists_handler")
    all_bonds = parse_md(raw)
    print(f"Total bonds: {len(all_bonds)}")

    test_ids = ["111025","111024","111020","113061","113060","113059","113058","113057","113056","113055","113054","113052","113050","113049","113048","113044"]
    print(f"Testing {len(test_ids)} bonds...")

    results = []
    config = MCConfig(n_paths=200, n_days=252, use_gpu=True)
    pricer = CCBPricer(config)

    for cb_id in test_ids:
        base_raw = call("ft_get_cb_base_data_handler", {"symbol_code": cb_id})
        base_rows = parse_md(base_raw)
        if not base_rows: print(f"  {cb_id}: no base data"); continue
        b = base_rows[0]
        name = b.get("cb_name", "?")
        if "退市" in name: print(f"  {cb_id}: delisted"); continue
        conv_price = float(b.get("conversion_price", 0) or 0)
        if conv_price <= 0: print(f"  {cb_id}: no conv_price"); continue

        stock_id = None
        for ab in all_bonds:
            if ab.get("cb_id") == cb_id: stock_id = ab.get("stock_id",""); break
        if not stock_id: print(f"  {cb_id}: no stock_id"); continue

        stock_code = f"{stock_id}.XSHG" if stock_id.startswith(("6","5")) else f"{stock_id}.XSHE"
        candle_raw = call("daily_ohlc", {"symbol": stock_code, "type": "stock"})
        try: candles = json.loads(candle_raw) if candle_raw else {}; items = candles.get("items",[])
        except: continue
        if not items: print(f"  {cb_id}: no candle data"); continue

        closes = [float(i["close"]) for i in items]
        stock_price = closes[-1]
        vol = float(np.std(np.diff(np.log(closes)))*np.sqrt(252)) if len(closes) >= 20 else 0.3

        cb_close = float(b.get("cb_close", 0) or 0)
        mkt = cb_close if cb_close > 0 else stock_price * (100.0/conv_price) * 1.05

        call_pct, put_pct = 1.30, 0.70
        try:
            ct = float(b.get("call_trigger_price",0) or 0)
            if ct>0: call_pct = ct/conv_price
            pt = float(b.get("put_trigger_price",0) or 0)
            if pt>0: put_pct = pt/conv_price
        except: pass

        ccb = CCBParams(name=name, ticker=cb_id, stock_ticker=stock_id,
                        face_value=100.0, conversion_price=conv_price,
                        days_to_maturity=252*2, call_trigger_pct=call_pct,
                        put_trigger_pct=put_pct, down_trigger_pct=0.85,
                        put_price=100.0, redemption_price=108.0,
                        market_price=mkt, stock_price=stock_price,
                        volatility=vol, conversion_start_day=0, industry="通用")
        try:
            mc_price, _ = pricer.price_single(ccb, n_paths=200, step_days=63, seed=42)
            disc = (mc_price-mkt)/mkt if mkt>0 else -1
            print(f"  {cb_id} {name[:25]:25s} S={stock_price:7.2f} σ={vol:.3f} C={conv_price:6.2f} MKT={mkt:7.2f} MC={mc_price:7.2f} disc={disc*100:+6.1f}%")
            results.append({"id":cb_id,"name":name,"S":stock_price,"vol":vol,"C":conv_price,"MKT":mkt,"MC":mc_price,"disc":disc})
        except Exception as e:
            print(f"  {cb_id}: err {e}")
        time.sleep(0.05)

    if not results: print("No results"); return

    df = pd.DataFrame(results).sort_values("disc", ascending=False)
    print(f"\n{'='*60}")
    print("Top picks by discount:")
    for i,(_,r) in enumerate(df.iterrows()):
        if i >= 10: break
        print(f"  {i+1}. {r['id']} {r['name'][:30]:30s} disc={r['disc']*100:+.1f}% MC={r['MC']:.2f} MKT={r['MKT']:.2f}")
    n_pos = (df["disc"]>0).sum()
    print(f"\nSummary: {len(df)} bonds, {n_pos} positive discount, avg disc={df['disc'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
