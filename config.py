"""
Config for Convertible Bond Monte Carlo Pricing & Backtest System
Based on: Liu (2025) - Valuation Model of Chinese Convertible Bonds Based on Monte Carlo Simulation
arXiv:2409.06496
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MCConfig:
    """Monte Carlo simulation parameters (matching paper defaults)"""
    n_paths: int = 5000           # M: number of Monte Carlo paths (paper uses 5000)
    n_days: int = 252             # T: trading days per year
    rfr: float = 0.03             # Annual risk-free rate (will be calibrated daily)
    transaction_cost: float = 0.001  # 0.1% transaction cost per trade
    
    # Stock dynamics
    use_gpu: bool = True          # GPU acceleration for path simulation
    dtype: str = "float32"        # Use float32 for GPU memory efficiency
    
    # Regression bases (paper: {St, St², Ft, Ft², Yt, Yt², StFt, StYt, FtYt})
    n_bases: int = 9
    
    # Multi-regression intervals
    # Π₁: (k_t, ∞)  Π₂: (C_t, k_t]  Π₃: (p_t, C_t]  Π₄: (-∞, p_t]
    n_intervals: int = 4
    
    # Downward adjustment
    p_downward: float = 0.8       # Probability of downward adjustment when put triggered
    
    # Call/Put trigger thresholds (paper defaults)
    pF: float = 0.5               # Call trigger: proportion threshold
    pY: float = 1.0               # Put trigger: proportion threshold
    mc: int = 15                  # Call lookback days
    nc: int = 30                  # Call observation window
    mp: int = 30                  # Put lookback days
    np: int = 30                  # Put observation window

@dataclass
class BacktestConfig:
    """Backtest strategy parameters"""
    n_positions: int = 10         # Top N most undervalued CCBs to hold
    rebalance_freq: str = "daily" # Rebalance frequency
    start_date: str = "2023-02-18"
    end_date: str = "2023-07-17"
    benchmark: str = "double_low" # Benchmark strategy

@dataclass 
class DataConfig:
    """Data source configuration"""
    mcp_endpoint: str = "https://market.ft.tech/gateway/mcp"
    cache_dir: str = "./data_cache"
    test_n_bonds: int = 20        # Initial test with 20 bonds
