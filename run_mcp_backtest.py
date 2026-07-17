#!/usr/bin/env python3
"""
Real A-share Convertible Bond Backtest using FTShare-MCP data.
Connects to https://market.ft.tech/gateway/mcp for live market data,
runs the LSM Monte Carlo pricing, and evaluates the strategy on real bonds.
"""
import sys, os, json, time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import httpx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from config import MCConfig, BacktestConfig
from pricer import CCBPricer, CCBParams


class MCPClient:
    """FTShare MCP client with SSE transport support."""
    
    def __init__(self, endpoint: str = "https://market.ft.tech/gateway/mcp"):
        self.endpoint = endpoint
        self.session_id: Optional[str] = None
        self.client = httpx.Client(timeout=60, follow_redirects=True)
        self._initialize()
    
    def _initialize(self):
        """MCP handshake: initialize -> initialized."""
        # Step 1: Initialize
        init_resp = self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hermes-cb-backtest", "version": "1.0"}
        })
        
        if "result" in init_resp:
            svr = init_resp["result"].get("serverInfo", {})
            logger.info(f"MCP connected: {svr.get('name', 'unknown')} v{svr.get('version', '?')}")
        
        # Step 2: Send initialized notification
        self._call("notifications/initialized", _id=2)
    
    def _call(self, method: str, params: dict = None, _id: int = 1) -> dict:
        """Make a JSON-RPC call over MCP with SSE parsing."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": _id,
        }
        if params:
            payload["params"] = params
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        
        resp = self.client.post(self.endpoint, json=payload, headers=headers)
        
        # Extract session ID from response headers
        sid = resp.headers.get("mcp-session-id")
        if sid and not self.session_id:
            self.session_id = sid
            logger.debug(f"MCP session: {sid[:12]}...")
        
        # Parse SSE response
        text = resp.text
        result = None
        
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if not data_str:
                    continue
                try:
                    data = json.loads(data_str)
                    if "result" in data or "error" in data:
                        result = data
                except json.JSONDecodeError:
                    continue
        
        if result is None:
            # Try parsing as plain JSON
            try:
                result = json.loads(text.strip())
            except json.JSONDecodeError:
                return {"error": f"Parse failed: {text[:200]}"}
        
        if "error" in result:
            err = result["error"]
            if err.get("code") == -32601:
                pass  # Method not found notification is expected
            else:
                logger.warning(f"MCP error: {err}")
        
        return result
    
    def call_tool(self, name: str, arguments: dict = None) -> Any:
        """Call a specific MCP tool and extract result."""
        resp = self._call("tools/call", {
            "name": name,
            "arguments": arguments or {}
        })
        
        if "result" in resp:
            content = resp["result"].get("content", [])
            for item in content:
                if item.get("type") == "text":
                    try:
                        return json.loads(item["text"])
                    except (json.JSONDecodeError, TypeError):
                        return item["text"]
            return resp["result"]
        
        return resp
    
    def get_cb_list(self) -> List[Dict]:
        """Get full convertible bond list."""
        result = self.call_tool("ft_get_cb_lists_handler")
        if isinstance(result, dict):
            items = result.get("items", result.get("data", []))
            return items if isinstance(items, list) else []
        return result if isinstance(result, list) else []
    
    def get_cb_base_data(self, symbol_code: str) -> Dict:
        """Get basic info for a single convertible bond."""
        return self.call_tool("ft_get_cb_base_data_handler", {"symbol_code": symbol_code})
    
    def get_stock_candles(self, symbol: str, start_date: str = None, 
                          end_date: str = None, interval: str = "day",
                          adjustment: str = "qfq") -> List[Dict]:
        """Get stock K-line data."""
        args = {
            "symbol": symbol,
            "interval": interval,
            "adjustment": adjustment,
        }
        if start_date:
            args["start_date"] = start_date
        if end_date:
            args["end_date"] = end_date
        
        result = self.call_tool("ft_stock_candlesticks", args)
        if isinstance(result, dict):
            return result.get("items", result.get("data", []))
        return result if isinstance(result, list) else []


def parse_cb_params(raw: Dict) -> Optional[CCBParams]:
    """Parse raw MCP bond data into CCBParams."""
    try:
        # Extract fields from FTShare response structure
        inner = raw.get("data", raw)
        
        name = str(inner.get("bond_full_name", inner.get("name", "")))
        ticker = str(inner.get("symbol_code", inner.get("bond_code", "")))
        
        # Stock code (正股代码)
        stock_code = str(inner.get("stock_code", inner.get("underlying_code", "")))
        
        # Conversion price (转股价)
        conv_price = float(inner.get("conv_price", inner.get("conversion_price", 100)))
        
        # Face value
        face_value = float(inner.get("par_value", inner.get("face_value", 100)))
        
        # Trigger prices as % of conversion price
        call_trigger = float(inner.get("call_trigger_price", conv_price * 1.30))
        call_trigger_pct = call_trigger / conv_price if conv_price > 0 else 1.30
        
        put_trigger = float(inner.get("put_trigger_price", conv_price * 0.70))
        put_trigger_pct = put_trigger / conv_price if conv_price > 0 else 0.70
        
        down_trigger = float(inner.get("down_trigger_price", conv_price * 0.85))
        down_trigger_pct = down_trigger / conv_price if conv_price > 0 else 0.85
        
        # Put price
        put_price = float(inner.get("put_price", 100.0))
        
        # Redemption at maturity
        redemption = float(inner.get("maturity_redemption_price", 
                           inner.get("redemption_price", 108.0)))
        
        # Market price
        market_price = float(inner.get("close_price", inner.get("market_price", 0)))
        
        # Days to maturity
        maturity_date = str(inner.get("maturity_date", ""))
        days_mat = 252 * 2  # Default 2 years
        if maturity_date:
            try:
                mat_dt = pd.Timestamp(maturity_date)
                today = pd.Timestamp.now()
                days_mat = max(1, (mat_dt - today).days * 252 // 365)
            except:
                pass
        
        # Trigger conditions
        mc = int(inner.get("call_trigger_days", inner.get("mc", 15)))
        nc = int(inner.get("call_observation_days", inner.get("nc", 30)))
        mp = int(inner.get("put_trigger_days", inner.get("mp", 30)))
        np_ = int(inner.get("put_observation_days", inner.get("np", 30)))
        
        # Conversion start
        conv_start_str = str(inner.get("conv_start_date", ""))
        conv_start = 126  # Default 6 months
        if conv_start_str:
            try:
                conv_dt = pd.Timestamp(conv_start_str)
                today = pd.Timestamp.now()
                days_since = (today - conv_dt).days
                conv_start = max(0, days_since * 252 // 365)
            except:
                pass
        else:
            conv_start = 0  # Already in conversion period
        
        # ratings
        industry = str(inner.get("industry", inner.get("bond_industry", "通用")))
        
        ccb = CCBParams(
            name=name,
            ticker=ticker,
            stock_ticker=stock_code,
            face_value=face_value,
            conversion_price=conv_price,
            days_to_maturity=days_mat,
            call_trigger_pct=call_trigger_pct,
            put_trigger_pct=put_trigger_pct,
            down_trigger_pct=down_trigger_pct,
            call_mc=mc, call_nc=nc,
            put_mp=mp, put_np=np_,
            put_price=put_price,
            redemption_price=redemption,
            market_price=market_price,
            stock_price=0.0,  # Will be set from stock data
            volatility=0.3,    # Will be computed from stock data
            conversion_start_day=conv_start,
            industry=industry,
        )
        return ccb
    except Exception as e:
        logger.warning(f"Failed to parse bond: {e}")
        return None


def compute_volatility(prices: List[float], annualize: int = 252) -> float:
    """Compute annualized historical volatility from price series."""
    if len(prices) < 20:
        return 0.3
    log_rets = np.diff(np.log(np.array(prices[-252:] + [prices[-1]])))
    vol = np.std(log_rets) * np.sqrt(annualize)
    return max(0.05, min(vol, 1.5))  # Clamp to reasonable range


def main():
    logger.info("=" * 60)
    logger.info("A股可转债MC定价回测 — 基于FTShare真实数据")
    logger.info("=" * 60)
    
    # 1. Connect to MCP
    logger.info("连接到 FTShare MCP...")
    mcp = MCPClient()
    
    # 2. Get convertible bond list
    logger.info("获取可转债列表...")
    cb_list = mcp.get_cb_list()
    logger.info(f"  获取到 {len(cb_list)} 只可转债")
    
    if not cb_list:
        logger.error("无法获取可转债列表，请检查网络和权限")
        return
    
    # Show first few bonds
    for i, cb in enumerate(cb_list[:10]):
        logger.info(f"  [{i+1}] {json.dumps(cb, ensure_ascii=False)[:120]}")
    
    # 3. For each bond, get base data
    # Take first 20 for quick test, all 487 for full
    test_bonds = cb_list[:20]
    logger.info(f"\n获取前 {len(test_bonds)} 只转债的详细信息...")
    
    bonds_data = []
    for cb_raw in tqdm(test_bonds, desc="Fetching bond data"):
        code = cb_raw.get("symbol_code", cb_raw.get("bond_code", ""))
        if not code:
            continue
        
        try:
            base_data = mcp.get_cb_base_data(code)
            parsed = parse_cb_params(base_data)
            if parsed:
                bonds_data.append(parsed)
        except Exception as e:
            logger.warning(f"  {code}: {e}")
            continue
    
    logger.info(f"成功解析 {len(bonds_data)} 只转债参数")
    
    if not bonds_data:
        logger.error("未能解析任何转债数据")
        return
    
    # 4. Get stock price data and compute volatility
    logger.info("\n获取正股历史行情...")
    for bond in tqdm(bonds_data, desc="Fetching stock prices"):
        stock_code = bond.stock_ticker
        if not stock_code:
            bond.volatility = 0.3
            continue
        
        # Normalize stock code (e.g., "600000" -> "600000.XSHG")
        if not "." in stock_code:
            if stock_code.startswith(("6", "5")):
                stock_code = f"{stock_code}.XSHG"
            elif stock_code.startswith(("0", "2", "3")):
                stock_code = f"{stock_code}.XSHE"
            elif stock_code.startswith(("4", "8")):
                stock_code = f"{stock_code}.BJSE"
        
        try:
            candles = mcp.get_stock_candles(
                stock_code,
                interval="day",
                adjustment="qfq",  # 前复权
            )
            
            if candles and len(candles) >= 20:
                # Extract closing prices
                closes = []
                for c in candles:
                    close_val = c.get("close", c.get("CLOSE", 0))
                    if close_val:
                        closes.append(float(close_val))
                
                if closes:
                    bond.stock_price = closes[-1]
                    bond.volatility = compute_volatility(closes)
        except Exception as e:
            logger.warning(f"  {stock_code}: {e}")
            bond.volatility = 0.3
    
    # Log bond summary
    logger.info(f"\n转债摘要 ({len(bonds_data)} 只):")
    for bond in bonds_data[:5]:
        logger.info(f"  {bond.name}({bond.ticker}): S={bond.stock_price:.2f}, "
                   f"σ={bond.volatility:.3f}, C={bond.conversion_price:.2f}, "
                   f"市场价={bond.market_price:.2f}")
    
    # 5. Run MC pricing for all bonds
    logger.info(f"\n开始MC定价 ({len(bonds_data)} 只转债)...")
    mc_config = MCConfig(n_paths=200, n_days=252, use_gpu=True)
    pricer = CCBPricer(mc_config)
    
    results = []
    for bond in tqdm(bonds_data, desc="MC Pricing"):
        if bond.stock_price <= 0 or bond.volatility <= 0:
            results.append({"ticker": bond.ticker, "model_price": 0, "market_price": bond.market_price, "discount": -1})
            continue
        
        try:
            price, info = pricer.price_single(bond, n_paths=200, step_days=63, seed=42)
            disc = (price - bond.market_price) / bond.market_price if bond.market_price > 0 else -1
            results.append({
                "ticker": bond.ticker,
                "name": bond.name,
                "stock_price": bond.stock_price,
                "volatility": bond.volatility,
                "conv_price": bond.conversion_price,
                "model_price": price,
                "market_price": bond.market_price,
                "discount": disc,
            })
        except Exception as e:
            logger.warning(f"  {bond.ticker} pricing failed: {e}")
            results.append({"ticker": bond.ticker, "model_price": 0, "market_price": bond.market_price, "discount": -1})
    
    # 6. Output results
    df = pd.DataFrame(results)
    df = df.sort_values("discount", ascending=False)
    
    logger.info("\n" + "=" * 60)
    logger.info("MC定价结果 — Top 10 折价率最高的转债")
    logger.info("=" * 60)
    
    header = f"{'排名':<4} {'代码':<10} {'名称':<12} {'市场价':<8} {'模型价':<8} {'折价率(%)':<10}"
    logger.info(header)
    logger.info("-" * len(header))
    
    for i, row in df.head(10).iterrows():
        logger.info(f"{i+1:<4} {row['ticker']:<10} {row['name']:<12} "
                   f"{row['market_price']:<8.2f} {row['model_price']:<8.2f} "
                   f"{row['discount']*100:<10.2f}")
    
    # Summary statistics
    positive = (df["discount"] > 0).sum()
    logger.info(f"\n统计: 共{len(df)}只转债, {positive}只折价(discount>0), "
               f"平均折价率={df['discount'].mean()*100:.2f}%")
    
    # Save results
    out_dir = Path(__file__).parent / "output_mcp"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "mcp_pricing_results.csv", index=False)
    logger.info(f"\n结果已保存到 {out_dir / 'mcp_pricing_results.csv'}")
    
    return df


if __name__ == "__main__":
    main()
