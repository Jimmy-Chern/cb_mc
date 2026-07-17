#!/usr/bin/env python3
"""A股可转债 MC定价 — FTShare MCP 真实数据, 完整版"""
import sys, os, json, time
from pathlib import Path
import httpx, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from config import MCConfig
from pricer import CCBPricer, CCBParams

EP = "https://market.ft.tech/gateway/mcp"
H = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def parse_md(text):
    """Parse markdown table into list of dicts."""
    rows = []; hdr = None
    for l in text.split("\n"):
        if not l.startswith("|"): continue
        cols = [x.strip() for x in l.split("|")[1:-1]]
        if "---" in l: continue
        if hdr is None: hdr = cols; continue
        if hdr: rows.append(dict(zip(hdr, cols)))
    return rows


def main():
    c = httpx.Client(timeout=30)
    r = c.post(EP, json={"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"cb","version":"1"}},"id":1}, headers=H)
    sid = r.headers.get("mcp-session-id","")
    c.post(EP, json={"jsonrpc":"2.0","method":"notifications/initialized","id":2}, headers={**H,"mcp-session-id":sid})
    print(f"MCP: {sid[:12]}...")

    def mcp(tool, args=None):
        r = c.post(EP, json={"jsonrpc":"2.0","method":"tools/call","params":{"name":tool,"arguments":args or {}},"id":int(time.time()*1000)%100000}, headers={**H,"mcp-session-id":sid})
        for line in r.text.split("\n"):
            if line.startswith("data:") and '"text"' in line:
                try:
                    d = json.loads(line[5:].strip())
                    for ct in d.get("result",{}).get("content",[]):
                        if ct.get("type")=="text": return ct["text"]
                except: pass
        return ""

    raw = mcp("ft_get_cb_lists_handler")
    all_bonds = parse_md(raw)
    print(f"Bond list: {len(all_bonds)} bonds")
    candidates = sorted(all_bonds, key=lambda x: x["cb_id"], reverse=True)[:100]
    print(f"Candidates: {len(candidates)}")

    results = []
    config = MCConfig(n_paths=200, n_days=252, use_gpu=True)
    pricer = CCBPricer(config)

    for bond_info in candidates:
        cb_id = bond_info["cb_id"]
        stock_id = bond_info["stock_id"]
        cb_sym = f"{cb_id}.XSHG" if cb_id.startswith("11") else f"{cb_id}.XSHE"
        cb_raw = mcp("daily_ohlc", {"symbol": cb_sym, "type": "stock"})
        try:
            cb_data = json.loads(cb_raw) if cb_raw else {}
            cb_items = cb_data.get("items", [])
        except: continue
        if not cb_items: continue
        cb_close = float(cb_items[-1]["close"])
        cb_vol = int(cb_items[-1].get("volume", 0))
        if cb_vol < 100000: continue

        base_raw = mcp("ft_get_cb_base_data_handler", {"symbol_code": cb_id})
        base_rows = parse_md(base_raw)
        if not base_rows: continue
        bd = base_rows[0]
        name = bd.get("cb_name", "?")
        if "退市" in name: continue
        conv_price = float(bd.get("conversion_price", 0) or 0)
        if conv_price <= 0: continue

        st_sym = f"{stock_id}.XSHG" if stock_id.startswith(("6","5")) else f"{stock_id}.XSHE"
        st_raw = mcp("daily_ohlc", {"symbol": st_sym, "type": "stock"})
        try:
            st_data = json.loads(st_raw) if st_raw else {}
            st_items = st_data.get("items", [])
        except: continue
        if not st_items: continue
        closes = [float(i["close"]) for i in st_items]
        stock_price = closes[-1]
        if len(closes) >= 20:
            lr = np.diff(np.log(closes))
            vol = float(np.std(lr) * np.sqrt(252))
            vol = max(0.05, min(vol, 1.5))
        else:
            vol = 0.3

        ccb = CCBParams(name=name, ticker=cb_id, stock_ticker=stock_id,
                        face_value=100.0, conversion_price=conv_price,
                        days_to_maturity=252*2, call_trigger_pct=1.30,
                        put_trigger_pct=0.70, down_trigger_pct=0.85,
                        put_price=100.0, redemption_price=108.0,
                        market_price=cb_close, stock_price=stock_price,
                        volatility=vol, conversion_start_day=0, industry="通用")
        try:
            mc_price, info = pricer.price_single(ccb, n_paths=200, step_days=63, seed=42)
            disc = (mc_price - cb_close) / cb_close
            cv = (100.0 / conv_price) * stock_price
            print(f"{cb_id} {name[:20]:20s} S={stock_price:7.2f} σ={vol:.3f} C={conv_price:6.2f} "
                  f"CV={cv:7.2f} MKT={cb_close:7.2f} MC={mc_price:7.2f} disc={disc*100:+6.1f}%")
            results.append({"id":cb_id,"name":name,"S":stock_price,"vol":vol,"C":conv_price,
                          "CV":cv,"MKT":cb_close,"MC":mc_price,"disc":disc,"cb_vol":cb_vol})
        except Exception as e:
            print(f"{cb_id}: err {e}")
        time.sleep(0.05)
        if len(results) >= 30: break

    if not results:
        print("No bonds priced"); return

    df = pd.DataFrame(results).sort_values("disc", ascending=False)
    print(f"\n{'='*70}")
    print(f"A股可转债 MC定价结果 ({len(df)}只)")
    print(f"{'='*70}")
    print(f"{'排':<2} {'代码':<8} {'名称':<22} {'正股':<7} {'转股价':<7} {'转股价值':<8} {'市场价':<7} {'MC价':<7} {'折价%':<8}")
    print("-"*70)
    for i,(_,r) in enumerate(df.iterrows()):
        tag = " ★" if i < 10 else ""
        print(f"{i+1:<2} {r['id']:<8} {r['name'][:22]:<22} {r['S']:<7.2f} {r['C']:<7.2f} "
              f"{r['CV']:<8.2f} {r['MKT']:<7.2f} {r['MC']:<7.2f} {r['disc']*100:+7.1f}%{tag}")

    n_pos = (df["disc"] > 0).sum()
    print(f"\n统计: {len(df)}只, {n_pos}只折价, 平均{df['disc'].mean()*100:+.1f}%, 中位数{df['disc'].median()*100:+.1f}%")
    print(f"\n★ 买入信号 (Top 10):")
    for i,(_,r) in enumerate(df.head(10).iterrows()):
        print(f"  {i+1}. {r['id']} {r['name'][:30]} 折价{r['disc']*100:+.1f}% (MC={r['MC']:.1f} MKT={r['MKT']:.1f})")

    out = Path(__file__).parent / "output_mcp"
    out.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    df.to_csv(out / f"real_pricing_{ts}.csv", index=False)
    print(f"\nSaved: {out}/real_pricing_{ts}.csv")


if __name__ == "__main__":
    main()
