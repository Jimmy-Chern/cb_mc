"""
GPU-accelerated Monte Carlo stock path simulation using PyTorch CUDA.

Generates M paths x T days of geometric Brownian motion:
    dS/S = (r - q) dt + σ dW
    S_t = S_{t-1} * exp((r - q - σ²/2) + σ * Z_t),  Z_t ~ N(0,1)
"""

import torch
import numpy as np
from typing import Tuple, Optional
from config import MCConfig


class StockPathSimulator:
    """GPU-accelerated geometric Brownian motion simulator for stock paths."""
    
    def __init__(self, config: MCConfig):
        self.config = config
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32 if config.dtype == "float32" else torch.float64
        
    def simulate_paths(
        self,
        S0: float,
        sigma: float,
        r: float,
        q: float = 0.0,
        n_days: Optional[int] = None,
        n_paths: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Simulate stock price paths using geometric Brownian motion.
        
        Args:
            S0: Initial stock price
            sigma: Annualized volatility
            r: Risk-free rate (annual, continuous compounding)
            q: Dividend yield (annual)
            n_days: Number of trading days to simulate
            n_paths: Number of Monte Carlo paths
            seed: Random seed for reproducibility
            
        Returns:
            Tensor of shape (n_paths, n_days+1) with stock prices.
            Index 0 = initial price S0.
        """
        n_paths = n_paths or self.config.n_paths
        n_days = n_days or self.config.n_days
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Daily parameters
        dt = 1.0 / 252  # daily step
        daily_r = r / 252
        daily_q = q / 252
        daily_sigma = sigma / np.sqrt(252)
        
        # Generate random normal increments: (n_paths, n_days)
        Z = torch.randn(n_paths, n_days, device=self.device, dtype=self.dtype)
        
        # Log-return per day: (r - q - σ²/2)*dt + σ*sqrt(dt)*Z
        drift = (daily_r - daily_q - 0.5 * daily_sigma ** 2)
        log_returns = drift + daily_sigma * Z
        
        # Cumulative sum + exponentiate
        log_prices = torch.cumsum(log_returns, dim=1)
        log_prices = torch.cat([
            torch.zeros(n_paths, 1, device=self.device, dtype=self.dtype),
            log_prices
        ], dim=1)
        
        prices = S0 * torch.exp(log_prices)
        return prices
    
    def simulate_batch(
        self,
        S0_batch: torch.Tensor,
        sigma_batch: torch.Tensor,
        r_batch: torch.Tensor,
        q_batch: torch.Tensor,
        n_days: Optional[int] = None,
        n_paths: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Batch-simulate paths for multiple stocks simultaneously.
        
        Args:
            S0_batch: (B,) initial prices for B bonds
            sigma_batch: (B,) volatilities
            r_batch: (B,) risk-free rates
            q_batch: (B,) dividend yields
            n_days: Trading days
            n_paths: Paths per bond
            
        Returns:
            Tensor of shape (B, n_paths, n_days+1)
        """
        B = len(S0_batch)
        n_paths = n_paths or self.config.n_paths
        n_days = n_days or self.config.n_days
        
        if seed is not None:
            torch.manual_seed(seed)
        
        dt = 1.0 / 252
        daily_r = r_batch[:, None, None].to(self.device, self.dtype) / 252  # (B, 1, 1)
        daily_q = q_batch[:, None, None].to(self.device, self.dtype) / 252
        daily_sigma = sigma_batch[:, None, None].to(self.device, self.dtype) / np.sqrt(252)  # (B, 1, 1)
        
        # Generate shared random numbers for efficiency
        Z = torch.randn(B, n_paths, n_days, device=self.device, dtype=self.dtype)
        
        drift = daily_r - daily_q - 0.5 * daily_sigma ** 2
        log_returns = drift + daily_sigma * Z
        
        log_prices = torch.cumsum(log_returns, dim=2)
        log_prices = torch.cat([
            torch.zeros(B, n_paths, 1, device=self.device, dtype=self.dtype),
            log_prices
        ], dim=2)
        
        prices = S0_batch[:, None, None].to(self.device, self.dtype) * torch.exp(log_prices)
        return prices


def historical_volatility(prices: np.ndarray, window: int = 252) -> float:
    """Calculate annualized historical volatility from daily prices."""
    returns = np.diff(np.log(prices))
    daily_vol = np.std(returns[-window:]) if len(returns) >= window else np.std(returns)
    return daily_vol * np.sqrt(252)


def test_simulator():
    """Quick test of the simulator."""
    config = MCConfig(n_paths=1000, n_days=252)
    sim = StockPathSimulator(config)
    
    # Test single stock
    paths = sim.simulate_paths(S0=100.0, sigma=0.3, r=0.03, seed=42)
    print(f"Single stock: shape={paths.shape}, device={paths.device}")
    print(f"  S0={paths[0,0].item():.2f}, mean final={paths[:,-1].mean().item():.2f}")
    
    # Test batch
    S0_batch = torch.tensor([100.0, 50.0, 200.0])
    sigma_batch = torch.tensor([0.3, 0.25, 0.35])
    r_batch = torch.tensor([0.03, 0.03, 0.03])
    q_batch = torch.tensor([0.0, 0.0, 0.0])
    
    batch_paths = sim.simulate_batch(S0_batch, sigma_batch, r_batch, q_batch, seed=42)
    print(f"Batch: shape={batch_paths.shape}, device={batch_paths.device}")
    print(f"  Stock 0 final mean={batch_paths[0,:,-1].mean().item():.2f}")
    print(f"  Stock 1 final mean={batch_paths[1,:,-1].mean().item():.2f}")
    print(f"  Stock 2 final mean={batch_paths[2,:,-1].mean().item():.2f}")


if __name__ == "__main__":
    test_simulator()
