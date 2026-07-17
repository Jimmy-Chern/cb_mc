#!/usr/bin/env python3
"""
A股可转债 MC定价 一年历史回测
数据源: FTShare MCP (转债参数) + yfinance (正股历史行情)
回测期: 2025-07-13 ~ 2026-07-13 (约250个交易日)
"""
import sys, os, json, time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np, pandas as pd
import httpx, yfinance as yf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import MCConfig
from pricer import CCBPricer, CCBParams

EP = "https://market.ft.tech/gateway/mcp"
H = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def parse_md(text):
    rows = []; hdr = None
    for l in text.split("\n"):
        if not l.startswith("|"): continue
        cols = [x.strip() for x in l.split("|")[1:-1]]
        if "---" in l: continue
        if hdr is None: hdr = cols; continue
        if hdr: rows.append(dict(zip(hdr, cols)))
    return rows


def init_mcp():
    c = httpx.Client(timeout=30)
    r = c.post(EP, json={"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"bt","version":"1"}},"id":1}, headers=H)
    sid = r.headers.get("mcp-session-id","")
    c.post(EP, json={"jsonrpc":"2.0","method":"notifications/initialized","id":2}, headers={**H,"mcp-session-id":sid})
    return c, sid

def mcp_call(c, sid, tool, args=None):
    r = c.post(EP, json={"jsonrpc":"2.0","method":"tools/call","params":{"name":tool,"arguments":args or {}},"id":int(time.time()*1000)%100000}, headers={**H,"mcp-session-id":sid})
    for line in r.text.split("\n"):
        if line.startswith("data:") and '"text"' in line:
            try:
                d = json.loads(line[5:].strip())
                for ct in d.get("result",{}).get("content",[]):
                    if ct.get("type")=="text": return ct["text"]
            except: pass
    return ""


def main():
    print("=" * 60)
    print("A股可转债 MC定价 一年历史回测")
    print("数据: FTShare MCP + yfinance")
    print("回测期: 1年 (约250交易日)")
    print("=" * 60)
    
    # 1. Get bond list + parameters from MCP
    print("\n[1/4] 获取可转债列表和参数...")
    c, sid = init_mcp()
    raw = mcp_call(c, sid, "ft_get_cb_lists_handler")
    all_bonds = parse_md(raw)
    print(f"  共 {len(all_bonds)} 只转债 (含历史)")

    # Filter to bonds with data
    bonds_config = []
    for bond_info in tqdm(all_bonds, desc="获取转债参数"):
        cb_id = bond_info["cb_id"]
        stock_id = bond_info["stock_id"]
        
        base_raw = mcp_call(c, sid, "ft_get_cb_base_data_handler", {"symbol_code": cb_id})
        base_rows = parse_md(base_raw)
        if not base_rows: continue
        bd = base_rows[0]
        name = bd.get("cb_name","?")
        if "退市" in name: continue
        
        conv_price = float(bd.get("conversion_price",0) or 0)
        if conv_price <= 0: continue
        
        # Get current CB market price
        cb_sym = f"{cb_id}.XSHG" if cb_id.startswith("11") else f"{cb_id}.XSHE"
        cb_raw = mcp_call(c, sid, "daily_ohlc", {"symbol": cb_sym, "type": "stock"})
        try:
            cb_data = json.loads(cb_raw) if cb_raw else {}
            cb_items = cb_data.get("items",[])
            if not cb_items: continue
            cb_close = float(cb_items[-1]["close"])
        except: continue
        
        call_pct, put_pct = 1.30, 0.70
        try:
            ct = float(bd.get("call_trigger_price",0) or 0)
            if ct>0: call_pct = ct/conv_price
            pt = float(bd.get("put_trigger_price",0) or 0)
            if pt>0: put_pct = pt/conv_price
        except: pass
        
        bonds_config.append({
            "cb_id": cb_id, "stock_id": stock_id, "name": name,
            "conv_price": conv_price, "call_pct": call_pct, "put_pct": put_pct,
            "cb_close": cb_close,
        })
        time.sleep(0.02)
    
    print(f"  有效转债: {len(bonds_config)} 只")
    
    # 2. Download historical stock data from yfinance
    print("\n[2/4] 下载正股历史行情 (yfinance)...")
    
    # Build yfinance tickers (Chinese stocks need .SS for Shanghai, .SZ for Shenzhen)
    yf_tickers = []
    for bc in bonds_config:
        sid = bc["stock_id"]
        if sid.startswith("6"): yf_tickers.append(f"{sid}.SS")
        elif sid.startswith(("0","2","3")): yf_tickers.append(f"{sid}.SZ")
        else: yf_tickers.append(f"{sid}.SS")
    
    # Download in batches (yfinance limits)
    end_date = "2026-07-13"
    start_date = "2025-07-13"
    
    stock_data = {}
    batch_size = 50
    for i in tqdm(range(0, len(yf_tickers), batch_size), desc="yfinance下载"):
        batch = yf_tickers[i:i+batch_size]
        try:
            data = yf.download(batch, start=start_date, end=end_date, progress=False, auto_adjust=True)
            for j, sym in enumerate(batch):
                bc = bonds_config[i+j]
                if len(batch) == 1:
                    closes = data["Close"]
                else:
                    closes = data["Close"][sym] if sym in data["Close"].columns else None
                if closes is not None and len(closes.dropna()) >= 20:
                    stock_data[bc["cb_id"]] = closes.dropna()
        except Exception as e:
            print(f"  yfinance batch failed: {e}")
            continue
    
    # Also download CB close prices
    cb_tickers = []
    for bc in bonds_config:
        if bc["cb_id"].startswith("11"): cb_tickers.append(f"{bc['cb_id']}.SS")
        else: cb_tickers.append(f"{bc['cb_id']}.SZ")
    
    cb_data = {}
    for i in tqdm(range(0, len(cb_tickers), batch_size), desc="yfinance转债行情"):
        batch = cb_tickers[i:i+batch_size]
        try:
            data = yf.download(batch, start=start_date, end=end_date, progress=False, auto_adjust=True)
            for j, sym in enumerate(batch):
                bc = bonds_config[i+j]
                if len(batch) == 1:
                    closes = data["Close"]
                else:
                    closes = data["Close"][sym] if sym in data["Close"].columns else None
                if closes is not None and len(closes.dropna()) > 0:
                    cb_data[bc["cb_id"]] = closes.dropna()
        except: continue
    
    # Filter bonds with both stock and CB price history
    valid_bonds = []
    for bc in bonds_config:
        if bc["cb_id"] in stock_data and bc["cb_id"] in cb_data:
            valid_bonds.append(bc)
    
    print(f"  有完整数据的转债: {len(valid_bonds)} 只")
    
    if len(valid_bonds) < 10:
        print("  数据不足, 无法回测!")
        return
    
    # 3. Build aligned trading calendar
    print("\n[3/4] 构建交易日历...")
    # Use the stock data from the first valid bond to get trading dates
    first_s = stock_data[valid_bonds[0]["cb_id"]]
    all_dates = sorted(set(first_s.index.tolist()))
    
    # Align all price series
    price_align = {}
    for bc in valid_bonds:
        cb_id = bc["cb_id"]
        s_prices = stock_data[cb_id].reindex(all_dates).ffill()
        c_prices = cb_data[cb_id].reindex(all_dates).ffill()
        if s_prices.dropna().iloc[0] > 0 and c_prices.dropna().iloc[0] > 0:
            price_align[cb_id] = {"stock": s_prices, "cb": c_prices}
    
    # Get common trading dates (at least 20 bonds have data)
    valid_dates = []
    for d in all_dates[252:]:  # Need 252 days for vol calculation
        count = 0
        for cb_id, prices in price_align.items():
            if d in prices["stock"].index and pd.notna(prices["stock"].get(d, np.nan)):
                count += 1
        if count >= 10:
            valid_dates.append(d)
    
    print(f"  有效交易日: {len(valid_dates)} 天")
    
    # 4. Backtest
    print(f"\n[4/4] 运行MC定价回测 ({len(valid_dates)}天)...")
    
    config = MCConfig(n_paths=100, n_days=252, use_gpu=True)
    pricer = CCBPricer(config)
    
    portfolio_values = [1.0]
    daily_returns = []
    position_log = []
    prev_positions = set()
    tc = 0.001  # 0.1% transaction cost
    
    for day_idx, today in enumerate(tqdm(valid_dates, desc="回测进度")):
        today_dt = today if isinstance(today, datetime) else pd.Timestamp(today)
        
        # Price all bonds with current data
        bond_prices = []
        for bc in valid_bonds:
            cb_id = bc["cb_id"]
            if cb_id not in price_align: continue
            
            prices = price_align[cb_id]
            if today not in prices["stock"].index: continue
            
            stock_p = float(prices["stock"].get(today, np.nan))
            cb_p = float(prices["cb"].get(today, np.nan))
            if pd.isna(stock_p) or pd.isna(cb_p) or stock_p <= 0 or cb_p <= 0:
                continue
            
            # Historical volatility (last 252 days up to yesterday)
            hist_s = prices["stock"].loc[:today]
            if len(hist_s) >= 252:
                hist_s = hist_s.iloc[-252:]
            if len(hist_s) >= 20:
                log_rets = np.diff(np.log(hist_s.values))
                vol = float(np.std(log_rets) * np.sqrt(252))
                vol = max(0.05, min(vol, 1.5))
            else:
                vol = 0.3
            
            bond_prices.append({
                "cb_id": cb_id, "name": bc["name"], "stock_p": stock_p,
                "cb_p": cb_p, "vol": vol, "conv_price": bc["conv_price"],
                "call_pct": bc["call_pct"], "put_pct": bc["put_pct"],
            })
        
        if len(bond_prices) < 10: continue
        
        # MC pricing
        discounts = []
        for bp in bond_prices:
            ccb = CCBParams(
                name=bp["name"], ticker=bp["cb_id"], stock_ticker="",
                face_value=100.0, conversion_price=bp["conv_price"],
                days_to_maturity=252*2, call_trigger_pct=bp["call_pct"],
                put_trigger_pct=bp["put_pct"], down_trigger_pct=0.85,
                put_price=100.0, redemption_price=108.0,
                market_price=bp["cb_p"], stock_price=bp["stock_p"],
                volatility=bp["vol"], conversion_start_day=0, industry="通用",
            )
            try:
                mc_price, _ = pricer.price_single(ccb, n_paths=100, step_days=63, seed=42)
                disc = (mc_price - bp["cb_p"]) / bp["cb_p"]
            except:
                disc = -1
            discounts.append((bp, disc))
        
        # Select top 10
        discounts.sort(key=lambda x: x[1], reverse=True)
        top10 = [d[0] for d in discounts[:10]]
        top10_ids = set(b["cb_id"] for b in top10)
        
        # Compute next-day returns
        if day_idx + 1 < len(valid_dates):
            tomorrow = valid_dates[day_idx + 1]
            position_returns = []
            for bp in top10:
                prices = price_align[bp["cb_id"]]
                if tomorrow in prices["cb"].index:
                    tmr_p = float(prices["cb"].get(tomorrow, np.nan))
                    if pd.notna(tmr_p) and tmr_p > 0:
                        ret = (tmr_p - bp["cb_p"]) / bp["cb_p"]
                        position_returns.append(ret)
            
            if position_returns:
                port_ret = np.mean(position_returns)
                turnover = len(top10_ids - prev_positions) / 10
                port_ret -= tc * turnover
                prev_positions = top10_ids
            else:
                port_ret = 0
        else:
            port_ret = 0
        
        daily_returns.append(port_ret)
        portfolio_values.append(portfolio_values[-1] * (1 + port_ret))
        
        if day_idx % 50 == 0:
            cum = (portfolio_values[-1] - 1) * 100
            print(f"  Day {day_idx}: cum_return={cum:.2f}%")
    
    # 5. Results
    ret_series = pd.Series(daily_returns, index=valid_dates[:len(daily_returns)])
    cum_ret = portfolio_values[-1] - 1
    ann_ret = (1+cum_ret)**(252/len(daily_returns)) - 1 if daily_returns else 0
    ann_vol = ret_series.std() * np.sqrt(252) if len(ret_series) > 1 else 0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum_series = (1+ret_series).cumprod()
    max_dd = (cum_series / cum_series.expanding().max() - 1).min()
    win_rate = (ret_series > 0).mean()
    
    # Benchmark: equal-weight all CBs
    bench_returns = []
    for day_idx, today in enumerate(valid_dates[:len(daily_returns)]):
        if day_idx + 1 < len(valid_dates):
            tomorrow = valid_dates[day_idx + 1]
            day_rets = []
            for bc in valid_bonds[:30]:  # First 30 as benchmark sample
                if bc["cb_id"] in price_align:
                    prices = price_align[bc["cb_id"]]
                    if today in prices["cb"].index and tomorrow in prices["cb"].index:
                        p_today = float(prices["cb"].get(today, np.nan))
                        p_tmr = float(prices["cb"].get(tomorrow, np.nan))
                        if pd.notna(p_today) and pd.notna(p_tmr) and p_today > 0:
                            day_rets.append((p_tmr-p_today)/p_today)
            bench_returns.append(np.mean(day_rets) if day_rets else 0)
        else:
            bench_returns.append(0)
    
    bench_cum = (1+pd.Series(bench_returns)).prod() - 1 if bench_returns else 0
    
    print("\n" + "=" * 60)
    print("A股可转债 MC定价 一年回测结果")
    print("=" * 60)
    print(f"回测期: {valid_dates[0].strftime('%Y-%m-%d') if valid_dates else 'N/A'} ~ "
          f"{valid_dates[-1].strftime('%Y-%m-%d') if valid_dates else 'N/A'}")
    print(f"交易日: {len(daily_returns)} 天")
    print(f"转债数量: {len(valid_bonds)} 只 (每日定价)")
    print()
    print(f"  {'指标':<20} {'LSM策略':<15} {'等权基准':<15}")
    print(f"  {'累计收益':<20} {cum_ret*100:>14.2f}% {bench_cum*100:>14.2f}%")
    print(f"  {'年化收益':<20} {ann_ret*100:>14.2f}% {'—':>15}")
    print(f"  {'年化波动':<20} {ann_vol*100:>14.2f}% {'—':>15}")
    print(f"  {'夏普比率':<20} {sharpe:>14.2f} {'—':>15}")
    print(f"  {'最大回撤':<20} {max_dd*100:>14.2f}% {'—':>15}")
    print(f"  {'日胜率':<20} {win_rate*100:>14.1f}% {'—':>15}")
    
    # Save
    out = Path(__file__).parent / "output_backtest"
    out.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    pd.DataFrame({
        "date": [str(d) for d in valid_dates[:len(daily_returns)]],
        "daily_return": daily_returns,
        "cumulative": np.array(portfolio_values[1:]),
    }).to_csv(out / f"backtest_{ts}.csv", index=False)
    
    print(f"\n结果保存: {out}/backtest_{ts}.csv")

if __name__ == "__main__":
    main()
