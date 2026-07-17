"""
Data fetcher for Chinese convertible bonds using FTShare-MCP.
Endpoints:
  - ft_get_cb_lists_handler: Get list of convertible bonds
  - ft_get_cb_base_data_handler: Get bond basic info
  - daily_ohlc: Get daily stock price data
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import requests
from loguru import logger

from config import DataConfig


@dataclass
class BondBasicInfo:
    """Basic information about a convertible bond."""
    ticker: str = ""
    name: str = ""
    stock_ticker: str = ""
    stock_name: str = ""
    
    # Bond parameters
    face_value: float = 100.0
    conversion_price: float = 0.0
    issue_date: str = ""
    listing_date: str = ""
    maturity_date: str = ""
    conversion_start_date: str = ""
    put_start_date: str = ""
    
    # Trigger prices
    call_trigger_pct: float = 1.30
    put_trigger_pct: float = 0.70
    down_trigger_pct: float = 0.85
    
    # Trigger conditions
    mc: int = 15    # call trigger days
    nc: int = 30    # call window days
    mp: int = 30    # put trigger days
    np: int = 30    # put window days
    
    # Prices
    put_price: float = 100.0
    redemption_price: float = 108.0
    
    # Ratings
    issuer_rating: str = ""
    bond_rating: str = ""
    industry: str = ""
    
    # Market data
    market_price: float = 0.0
    stock_price: float = 0.0
    premium_rt: float = 0.0  # Conversion premium rate


class FTShareMCPClient:
    """Client for FTShare MCP data service."""
    
    def __init__(self, config: DataConfig, cache_dir: Optional[str] = None):
        self.config = config
        self.endpoint = config.mcp_endpoint
        self.cache_dir = Path(cache_dir or config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        
        # Load credentials if available
        self._load_credentials()
    
    def _load_credentials(self):
        """Load MCP credentials from environment or config files."""
        cred_file = Path.home() / ".ftshare_mcp_credentials.json"
        if cred_file.exists():
            with open(cred_file) as f:
                creds = json.load(f)
                if "api_key" in creds:
                    self.session.headers["Authorization"] = f"Bearer {creds['api_key']}"
                if "token" in creds:
                    self.session.headers["X-API-Token"] = creds["token"]
        
        # Also check env vars
        if "FTSHARE_API_KEY" in os.environ:
            self.session.headers["Authorization"] = f"Bearer {os.environ['FTSHARE_API_KEY']}"
        if "FTSHARE_TOKEN" in os.environ:
            self.session.headers["X-API-Token"] = os.environ["FTSHARE_TOKEN"]
    
    def _call_mcp(self, tool_name: str, arguments: Dict = None) -> Dict:
        """Call an MCP tool."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            },
            "id": 1
        }
        
        cache_key = hashlib.md5(
            f"{tool_name}:{json.dumps(arguments, sort_keys=True)}".encode()
        ).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Check cache (1 hour TTL)
        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < 3600:
                try:
                    with open(cache_file) as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass
        
        # Make request
        try:
            resp = self.session.post(self.endpoint, json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            
            # Cache result
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"MCP call failed: {tool_name} - {e}")
            # Return cached data even if expired
            if cache_file.exists():
                with open(cache_file) as f:
                    return json.load(f)
            raise
    
    def get_cb_list(self) -> List[Dict]:
        """Get the full list of convertible bonds."""
        result = self._call_mcp("ft_get_cb_lists_handler")
        return self._parse_result(result)
    
    def get_cb_base_data(self, ticker: str) -> Dict:
        """Get basic data for a specific convertible bond."""
        result = self._call_mcp("ft_get_cb_base_data_handler", {"ticker": ticker})
        return self._parse_result(result)
    
    def get_daily_ohlc(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get daily OHLC data for a stock/CCB ticker."""
        args = {"ticker": ticker}
        if start_date:
            args["start_date"] = start_date
        if end_date:
            args["end_date"] = end_date
        
        result = self._call_mcp("daily_ohlc", args)
        data = self._parse_result(result)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            return pd.DataFrame()
        
        return df
    
    def _parse_result(self, result: Dict) -> Any:
        """Parse MCP result from JSON-RPC response."""
        if "result" in result:
            content = result["result"]
            if isinstance(content, dict) and "content" in content:
                for item in content["content"]:
                    if item.get("type") == "text":
                        try:
                            return json.loads(item["text"])
                        except (json.JSONDecodeError, TypeError):
                            return item["text"]
            return content
        if "error" in result:
            logger.error(f"MCP error: {result['error']}")
            return {}
        return result


class LocalDataProvider:
    """
    Fallback data provider using locally cached data.
    When MCP is unavailable, uses pre-saved data or generates synthetic data for testing.
    """
    
    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_test_bonds(self, n: int = 20) -> List[Dict]:
        """Get test bond data for pipeline validation. Generates up to N bonds."""
        import random
        random.seed(42)
        np.random.seed(42)
        
        industries = ["铁路运输", "航空运输", "建筑", "制造业", "汽车", "环保", "钢铁", 
                      "银行", "租赁", "券商", "电力", "机械", "化工", "证券", 
                      "有色金属", "基建", "医药", "电子", "食品饮料", "房地产",
                      "通信", "传媒", "计算机", "纺织服装", "农林牧渔", "商贸",
                      "家电", "轻工制造", "国防军工", "公用事业"]
        
        ratings = ["AAA", "AAA", "AA+", "AA+", "AA+", "AA", "AA", "AA", "AA-", "AA-"]
        
        bonds = []
        for i in range(n):
            # Generate diverse bond parameters
            stock_price = np.random.uniform(3, 50)
            conv_price = stock_price * np.random.uniform(0.7, 1.3)
            vol = np.random.uniform(0.18, 0.55)
            
            # Market price based on conversion value + premium/noise
            conv_value = (100.0 / conv_price) * stock_price
            market_price = conv_value * np.random.uniform(0.85, 1.30)
            market_price = max(market_price, 80.0)  # Floor near bond value
            
            # Trigger percentages vary
            call_pct = np.random.choice([1.20, 1.25, 1.30, 1.30, 1.30])
            put_pct = np.random.choice([0.65, 0.70, 0.70, 0.70, 0.75])
            down_pct = np.random.choice([0.80, 0.85, 0.85, 0.85, 0.90])
            
            # Maturity: 1-6 years
            days_mat = int(np.random.uniform(252, 252*6))
            
            # Redemption: 105-110
            redemption = np.random.choice([105, 106, 107, 108, 108, 108, 110])
            
            bonds.append({
                "ticker": f"{113000 + i:06d}" if i < 1000 else f"{120000 + i:06d}",
                "name": f"Test CB {i+1}",
                "stock_ticker": f"{600000 + i:06d}",
                "stock_name": f"Stock {i+1}",
                "face_value": 100.0,
                "conversion_price": round(conv_price, 2),
                "listing_date": "2021-01-15",
                "maturity_date": f"20{23 + (days_mat // 252):02d}-12-14",
                "conversion_start_date": "2021-06-18",
                "put_start_date": f"20{23 + (days_mat // 252) - 1:02d}-12-13",
                "call_trigger_pct": call_pct,
                "put_trigger_pct": put_pct,
                "down_trigger_pct": down_pct,
                "mc": 15, "nc": 30, "mp": 30, "np": 30,
                "put_price": 100.0,
                "redemption_price": redemption,
                "issuer_rating": random.choice(ratings),
                "bond_rating": random.choice(ratings),
                "industry": random.choice(industries),
                "market_price": round(market_price, 2),
                "stock_price": round(stock_price, 2),
                "volatility": round(vol, 4),
                "days_to_maturity": days_mat,
                "premium_rt": round((market_price - conv_value) / conv_value * 100, 2) if conv_value > 0 else 0,
            })
        
        return bonds


class DataManager:
    """Central data manager combining MCP client and local fallback."""
    
    def __init__(self, data_config: DataConfig):
        self.config = data_config
        self.mcp = FTShareMCPClient(data_config)
        self.local = LocalDataProvider(data_config.cache_dir)
    
    def fetch_bond_universe(self, use_mcp: bool = True) -> pd.DataFrame:
        """Fetch the full bond universe with basic info."""
        if use_mcp:
            try:
                logger.info("Fetching bond list from MCP...")
                bond_list = self.mcp.get_cb_list()
                if bond_list and len(bond_list) > 0:
                    logger.info(f"Got {len(bond_list)} bonds from MCP")
                    return pd.DataFrame(bond_list)
            except Exception as e:
                logger.warning(f"MCP fetch failed: {e}, falling back to local data")
        
        logger.info("Using local test data")
        bonds = self.local.get_test_bonds(n=self.config.test_n_bonds)
        return pd.DataFrame(bonds)
    
    def fetch_historical_prices(
        self, 
        stock_ticker: str, 
        start_date: str, 
        end_date: str,
        use_mcp: bool = True
    ) -> pd.DataFrame:
        """Fetch historical stock prices."""
        if use_mcp:
            try:
                df = self.mcp.get_daily_ohlc(stock_ticker, start_date, end_date)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Failed to fetch {stock_ticker}: {e}")
        
        # Generate synthetic historical data for testing
        return self._generate_synthetic_prices(start_date, end_date)
    
    def _generate_synthetic_prices(self, start_date: str, end_date: str, 
                                    S0: float = 100.0, sigma: float = 0.3) -> pd.DataFrame:
        """Generate synthetic price history for testing."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        dates = pd.bdate_range(start, end)
        n = len(dates)
        
        np.random.seed(hash(start_date) % 2**32)
        returns = np.random.normal(0.0002, sigma / np.sqrt(252), n)
        prices = S0 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            "date": dates,
            "open": prices * (1 + np.random.normal(0, 0.01, n)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.02, n))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.02, n))),
            "close": prices,
            "volume": np.random.randint(1e6, 1e8, n),
        })
