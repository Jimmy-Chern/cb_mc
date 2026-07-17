#!/usr/bin/env python3
"""
A股可转债MC定价回测 — FTShare MCP 真实数据版
"""
import sys, os, json, re, time
from pathlib import Path
from datetime import datetime

import httpx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from config import MCConfig
from pricer import CCBPricer, CCBParams


ENDPOINT = "https://market.ft.tech/gateway/mcp"
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


class MCPSession:
    def __init__(self):
        self.client = httpx.Client(timeout=60)
        self.sid = self._init()
    
    def _init(self):
        r = self.client.post(ENDPOINT, json={
            "jsonrpc":"2.0","method":"initialize",
            "params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"cb","version":"1"}},
            "id":1
        }, headers=HEADERS)
        sid = r.headers.get("mcp-session-id", "")
        self.client.post(ENDPOINT, json={"jsonrpc":"2.0","method":"notifications/initialized","id":2},
                        headers={**HEADERS, "mcp-session-id": sid})
        return sid
    
    def call(self, tool: str, args: dict = None) -> str:
        r = self.client.post(ENDPOINT, json={
            "jsonrpc":"2.0","method":"tools/call",
            "params":{"name": tool, "arguments": args or {}},
            "id": int(time.time()*1000) % 100000
        }, headers={**HEADERS, "mcp-session-id": self.sid})
        # Extract text from SSE
        for line in r.text.split("\n"):
            if line.startswith("data:") and '"text"' in line:
                try:
                    d = json.loads(line[5:].strip())
                    for c in d.get("result",{}).get("content",[]):
                        if c.get("type")=="text":
                            return c["text"]
                except: pass
        return ""


def parse_md_table(text: str) -> list:
    """Parse markdown table into list of dicts."""
    rows = []
    header = None
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.split("|")[1:-1]]
        if "---" in line:
            continue
        if header is None and "cb_id" in line.lower():
            header = cols
            continue
        if header:
            rows.append(dict(zip(header, cols)))
    return rows


def main():
    logger.info("=== A股可转债MC定价 — FTShare MCP ===")
    
    mcp = MCPSession()
    logger.info(f"MCP session: {mcp.sid[:12]}...")
    
    # 1. Get CB list
    logger.info("获取可转债列表...")
    raw_list = mcp.call("ft_get_cb_lists_handler")
    bonds_raw = parse_md_table(raw_list)
    logger.info(f"共 {len(bonds_raw)} 只可转债")
    
    # Show first few
    for b in bonds_raw[:5]:
        logger.info(f"  {b['cb_id']} {b['name'][:30]}... → 正股{b['stock_id']}")
    
    # 2. Get base data + stock prices for N bonds (20 for quick test)
    N_TEST = 20
    test_bonds = bonds_raw[:N_TEST]
    
    results = []
    
    for bond_info in tqdm(test_bonds, desc="获取转债数据"):
        cb_id = bond_info["cb_id"]
        stock_id = bond_info["stock_id"]
        name = bond_info["name"]
        
        # Get base data
        raw_base = mcp.call("ft_get_cb_base_data_handler", {"symbol_code": cb_id})
        
        # Get stock candles for volatility + current price
        stock_code = f"{stock_id}.XSHG" if stock_id.startswith(("6","5")) else f"{stock_id}.XSHE"
        raw_candles = mcp.call("ft_stock_candlesticks", {
            "symbol": stock_code, "interval": "day", "adjustment": "qfq"
        })
        
        # Parse base data from markdown table
        base_rows = parse_md_table(raw_base)
        base = {}
        if base_rows:
            base = {k.lower().replace(" ","_"): v for k,v in base_rows[0].items()}
        
        # Parse candles
        candles = {}
        if raw_candles:
            candle_rows = parse_md_table(raw_candles)
            if candle_rows:
                closes = []
                for cr in candle_rows:
                    close_val = float(cr.get("close", cr.get("收盘", 0)))
                    if close_val > 0:
                        closes.append(close_val)
                if closes:
                    candles["close"] = closes[-1]
                    if len(closes) >= 20:
                        log_rets = np.diff(np.log(closes[-252:]))
                        candles["volatility"] = float(np.std(log_rets) * np.sqrt(252))
                    else:
                        candles["volatility"] = 0.3
        
        stock_price = candles.get("close", 0)
        volatility = candles.get("volatility", 0.3)
        
        # Get key parameters from base data
        conv_price_str = base.get("conv_price", base.get("转股价", "100"))
        try:
            conv_price = float(conv_price_str.replace(",",""))
        except:
            conv_price = 100
        
        close_str = base.get("cb_close", base.get("close", "0"))
        try:
            market_price = float(close_str.replace(",",""))
        except:
            market_price = 100
        
        call_trig_str = base.get("call_trigger_price", base.get("强赎触发价", str(conv_price*1.3)))
        try:
            call_trigger = float(call_trig_str.replace(",",""))
        except:
            call_trigger = conv_price * 1.3
        
        put_trig_str = base.get("put_trigger_price", base.get("回售触发价", str(conv_price*0.7)))
        try:
            put_trigger = float(put_trig_str.replace(",",""))
        except:
            put_trigger = conv_price * 0.7
        
        call_pct = call_trigger / conv_price if conv_price > 0 else 1.30
        put_pct = put_trigger / conv_price if conv_price > 0 else 0.70
        
        # Build CCBParams
        ccb = CCBParams(
            name=name[:30],
            ticker=cb_id,
            stock_ticker=stock_id,
            face_value=100.0,
            conversion_price=conv_price,
            days_to_maturity=252*2,
            call_trigger_pct=call_pct,
            put_trigger_pct=put_pct,
            down_trigger_pct=0.85,
            put_price=100.0,
            redemption_price=108.0,
            market_price=market_price,
            stock_price=stock_price,
            volatility=volatility,
            conversion_start_day=0,
            industry="通用",
        )
        
        # Run MC pricing
        if stock_price > 0 and volatility > 0:
            config = MCConfig(n_paths=200, n_days=252, use_gpu=True)
            pricer = CCBPricer(config)
            try:
                model_price, info = pricer.price_single(ccb, n_paths=200, step_days=63, seed=42)
                discount = (model_price - market_price) / market_price if market_price > 0 else 0
            except Exception as e:
                model_price = 0
                discount = 0
        else:
            model_price = 0
            discount = 0
        
        results.append({
            "ticker": cb_id,
            "name": name[:40],
            "stock_id": stock_id,
            "stock_price": stock_price,
            "volatility": volatility,
            "conv_price": conv_price,
            "market_price": market_price,
            "model_price": model_price,
            "discount": discount,
            "call_pct": call_pct,
            "put_pct": put_pct,
        })
        
        time.sleep(0.1)  # Rate limit
    
    # Output results
    df = pd.DataFrame(results)
    df = df.sort_values("discount", ascending=False)
    
    logger.info("\n" + "="*70)
    logger.info("    A股可转债 MC定价结果 — Top 10 折价率")
    logger.info("="*70)
    logger.info(f"{'排':<2} {'代码':<8} {'转债名称':<25} {'市场价':<7} {'模型价':<7} {'折价率%':<8} {'σ':<5} {'正股价':<7}")
    logger.info("-"*70)
    
    for i, (_, row) in enumerate(df.head(10).iterrows()):
        logger.info(f"{i+1:<2} {row['ticker']:<8} {row['name'][:25]:<25} "
                   f"{row['market_price']:<7.2f} {row['model_price']:<7.2f} "
                   f"{row['discount']*100:<8.2f} {row['volatility']:<5.3f} {row['stock_price']:<7.2f}")
    
    n_pos = (df["discount"] > 0).sum()
    avg_disc = df["discount"].mean() * 100
    logger.info(f"\n共 {len(df)} 只转债, {n_pos} 只折价(discount>0), 平均折价率 {avg_disc:.2f}%")
    
    # Top 10 picks  
    top10 = df.head(10)
    logger.info(f"\n今日买入信号 (Top 10):")
    for i, (_, row) in enumerate(top10.iterrows()):
        logger.info(f"  {i+1}. {row['ticker']} {row['name'][:30]} (折价{row['discount']*100:.1f}%)")
    
    # Save
    out = Path(__file__).parent / "output_mcp"
    out.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(out / f"pricing_{timestamp}.csv", index=False)
    logger.info(f"\n结果保存: {out}/pricing_{timestamp}.csv")
    
    return df


if __name__ == "__main__":
    main()
