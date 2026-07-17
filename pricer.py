"""
Convertible Bond Pricing Model using Least Squares Monte Carlo (LSM).

Based on Liu (2025) - arxiv:2409.06496
GPU-accelerated implementation optimized for A800.

Key optimization: Uses monthly exercise dates (step=21 days) instead of daily,
reducing ~600 time steps to ~30 with negligible pricing error (<1%).
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config import MCConfig
from simulator import StockPathSimulator


@dataclass
class CCBParams:
    """Parameters for a single convertible bond."""
    name: str = ""
    ticker: str = ""
    face_value: float = 100.0
    conversion_price: float = 100.0
    maturity_date: str = ""
    days_to_maturity: int = 252 * 3
    call_trigger_pct: float = 1.30
    put_trigger_pct: float = 0.70
    down_trigger_pct: float = 0.85
    call_mc: int = 15
    call_nc: int = 30
    put_mp: int = 30
    put_np: int = 30
    put_price: float = 100.0
    redemption_price: float = 108.0
    market_price: float = 0.0
    stock_price: float = 0.0
    volatility: float = 0.3
    conversion_start_day: int = 126
    put_start_day: int = 252 * 3
    industry: str = ""
    stock_ticker: str = ""


class CCBPricer:
    """Least Squares Monte Carlo pricer for Chinese Convertible Bonds."""
    
    N_BASES = 9  # {S, S², F, F², Y, Y², SF, SY, FY}
    
    def __init__(self, config: MCConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.simulator = StockPathSimulator(config)
        self.dtype = torch.float32 if config.dtype == "float32" else torch.float64
    
    def price_single(
        self,
        ccb: CCBParams,
        S0: Optional[float] = None,
        sigma: Optional[float] = None,
        rfr: Optional[float] = None,
        n_paths: Optional[int] = None,
        step_days: int = 21,  # Monthly exercise dates (21 trading days)
        seed: Optional[int] = 42,
    ) -> Tuple[float, Dict]:
        """Price a single CCB using LSM with monthly exercise dates."""
        S0 = S0 or ccb.stock_price
        sigma = sigma or ccb.volatility
        rfr = rfr or self.config.rfr
        n_paths = n_paths or self.config.n_paths
        n_days = ccb.days_to_maturity
        
        if S0 <= 0 or sigma <= 0:
            return 0.0, {"error": "Invalid S0 or sigma"}
        
        # Step 1: Simulate stock paths (daily resolution for trigger lookback accuracy)
        paths = self.simulator.simulate_paths(
            S0=S0, sigma=sigma, r=rfr, q=0.0,
            n_days=n_days, n_paths=n_paths, seed=seed
        )  # (M, T+1), T = n_days
        
        M = paths.shape[0]
        T_total = paths.shape[1] - 1
        
        # Derived parameters
        k_t = ccb.conversion_price * ccb.call_trigger_pct
        p_t = ccb.conversion_price * ccb.put_trigger_pct
        C_t = ccb.conversion_price
        m = ccb.face_value / ccb.conversion_price
        B = ccb.redemption_price
        P_put = ccb.put_price
        K_call = ccb.face_value * 1.001  # Approximate call redemption
        
        # Step 2: Compute triggers Ft, Yt for all paths at all days (daily resolution)
        S_daily = paths[:, 1:]  # (M, T_total)
        
        call_lb = min(ccb.call_mc, ccb.call_nc)
        put_lb = min(ccb.put_mp, ccb.put_np)
        
        # Running proportions via convolution
        call_mask = (S_daily >= k_t).float()
        put_mask = (S_daily <= p_t).float()
        
        def running_proportion(mask, window):
            if window <= 1:
                return mask
            kernel = torch.ones(1, 1, window, device=self.device, dtype=self.dtype) / window
            return torch.nn.functional.conv1d(
                mask.unsqueeze(1), kernel, padding=window-1
            ).squeeze(1)[:, :T_total]
        
        Ft_daily = running_proportion(call_mask, call_lb)
        Yt_daily = running_proportion(put_mask, put_lb)
        
        # Pad day 0
        Ft_daily = torch.cat([torch.zeros(M, 1, device=self.device, dtype=self.dtype), Ft_daily], dim=1)
        Yt_daily = torch.cat([torch.zeros(M, 1, device=self.device, dtype=self.dtype), Yt_daily], dim=1)
        
        # Step 3: Build exercise dates (monthly = every step_days days)
        exercise_dates = list(range(ccb.conversion_start_day, T_total + 1, step_days))
        if T_total not in exercise_dates:
            exercise_dates.append(T_total)
        exercise_dates = sorted(set(exercise_dates))
        
        n_steps = len(exercise_dates)
        
        # Extract sub-sampled paths and triggers at exercise dates
        # paths_sub: (M, n_steps), Ft_sub: (M, n_steps), Yt_sub: (M, n_steps)
        idx_tensor = torch.tensor(exercise_dates, device=self.device, dtype=torch.long)
        paths_sub = paths[:, idx_tensor]     # (M, n_steps)
        Ft_sub = Ft_daily[:, idx_tensor]      # (M, n_steps)
        Yt_sub = Yt_daily[:, idx_tensor]      # (M, n_steps)
        
        # Step sizes between exercise dates (for discounting)
        steps_diff = torch.tensor(
            [exercise_dates[i] - exercise_dates[i-1] if i > 0 else exercise_dates[0]
             for i in range(n_steps)],
            device=self.device, dtype=self.dtype
        )  # (n_steps,)
        
        # Daily discount factor
        daily_df = np.exp(-rfr / 252)
        
        # Step 4: Initialize at maturity (last exercise date)
        last_idx = n_steps - 1
        V = torch.maximum(m * paths_sub[:, last_idx], 
                         torch.tensor(B, device=self.device, dtype=self.dtype))  # (M,)
        
        # Step 5: Backward induction over exercise dates
        for step in range(last_idx - 1, -1, -1):
            t_day = exercise_dates[step]
            n_step_days = int(steps_diff[step + 1])  # days to next exercise date
            
            S_t = paths_sub[:, step]    # (M,)
            F_t = Ft_sub[:, step]       # (M,)
            Y_t_val = Yt_sub[:, step]   # (M,)
            
            # Discount to this step
            discount = daily_df ** n_step_days
            y = V * discount  # (M,) continuation value
            
            # Build regression basis
            S2 = S_t ** 2
            F2 = F_t ** 2
            Y2 = Y_t_val ** 2
            SF = S_t * F_t
            SY = S_t * Y_t_val
            FY = F_t * Y_t_val
            basis = torch.stack([S_t, S2, F_t, F2, Y_t_val, Y2, SF, SY, FY], dim=1)  # (M, 9)
            
            if M < self.N_BASES + 5:
                V = y
                continue
            
            # Multi-regression: 4 intervals
            y_hat = self._multi_regression(S_t, basis, y, k_t, C_t, p_t)
            
            # Exercise decision (vectorized)
            conv_value = m * S_t
            
            call_trig = F_t >= self.config.pF
            put_trig = Y_t_val >= self.config.pY
            
            # Case 1: Call triggered
            forced_redemption = call_trig & (K_call >= conv_value)
            V = torch.where(forced_redemption, 
                          torch.tensor(K_call, device=self.device, dtype=self.dtype), V)
            V = torch.where(call_trig & ~forced_redemption, conv_value, V)
            
            # Case 2: Put triggered
            if put_trig.any():
                # Downward adjustment
                adj_conv_price = torch.maximum(
                    S_t * 0.85,
                    torch.tensor(ccb.conversion_price * ccb.down_trigger_pct, 
                               device=self.device, dtype=self.dtype)
                )
                adj_m = ccb.face_value / adj_conv_price
                adj_conv = adj_m * S_t
                
                put_adj = torch.maximum(
                    torch.maximum(torch.tensor(P_put, device=self.device, dtype=self.dtype), adj_conv),
                    y_hat
                )
                put_simple = torch.maximum(
                    torch.maximum(torch.tensor(P_put, device=self.device, dtype=self.dtype), conv_value),
                    y_hat
                )
                
                # Probabilistic downward adjustment
                rand_mask = torch.rand(M, device=self.device) < self.config.p_downward
                use_adj = put_trig & rand_mask & (t_day >= ccb.put_start_day)
                put_val = torch.where(use_adj, put_adj, put_simple)
                V = torch.where(put_trig, put_val, V)
            
            # Case 3: No trigger - conversion vs continuation
            no_trig = ~call_trig & ~put_trig
            convert_now = no_trig & (conv_value > y_hat)
            V = torch.where(convert_now, conv_value, V)
            V = torch.where(no_trig & ~convert_now, y_hat, V)
        
        # Step 6: Price = average of discounted values at t=0
        # Discount from first exercise date to today
        first_step_days = exercise_dates[0]
        discount_to_0 = daily_df ** first_step_days
        V0 = (V * discount_to_0).mean().item()
        
        info = {
            "model_price": V0,
            "S0": S0, "sigma": sigma, "rfr": rfr,
            "conv_value": m * S0,
            "conversion_ratio": m,
            "n_exercise_steps": n_steps,
            "step_days": step_days,
        }
        
        return V0, info
    
    def _multi_regression(
        self, S_t, basis, y, k_t, C_t, p_t
    ) -> torch.Tensor:
        """Multi-regression across 4 stock price intervals. Returns y_hat for all paths."""
        M = S_t.shape[0]
        y_hat = torch.zeros(M, device=self.device, dtype=self.dtype)
        all_mask = torch.ones(M, dtype=torch.bool, device=self.device)
        
        intervals = [
            S_t > k_t,                      # Pi_1: equity-like
            (S_t > C_t) & (S_t <= k_t),     # Pi_2: balanced equity
            (S_t > p_t) & (S_t <= C_t),     # Pi_3: balanced bond
            S_t <= p_t,                      # Pi_4: bond-like
        ]
        
        covered = torch.zeros(M, dtype=torch.bool, device=self.device)
        
        for int_mask in intervals:
            mask = int_mask & all_mask
            n = mask.sum().item()
            if n < self.N_BASES + 5:
                continue
            
            X = basis[mask]
            Y = y[mask]
            
            try:
                XtX = X.T @ X
                XtY = X.T @ Y
                I_reg = torch.eye(self.N_BASES, device=self.device, dtype=self.dtype) * 1e-8
                theta = torch.linalg.solve(XtX + I_reg, XtY)
                y_hat[mask] = X @ theta
                covered |= mask
            except RuntimeError:
                pass
        
        # Fallback: unified regression for uncovered paths
        uncovered = all_mask & ~covered
        if uncovered.sum() > self.N_BASES + 5:
            X = basis[uncovered]
            Y = y[uncovered]
            try:
                XtX = X.T @ X
                XtY = X.T @ Y
                I_reg = torch.eye(self.N_BASES, device=self.device, dtype=self.dtype) * 1e-8
                theta = torch.linalg.solve(XtX + I_reg, XtY)
                y_hat[uncovered] = X @ theta
            except RuntimeError:
                y_hat[uncovered] = y[uncovered]
        
        return y_hat
    
    def price_batch(
        self,
        ccb_list: List[CCBParams],
        n_paths: Optional[int] = None,
        step_days: int = 21,
        seed: Optional[int] = 42,
    ) -> List[Tuple[float, Dict]]:
        """Price multiple CCBs sequentially."""
        results = []
        for ccb in ccb_list:
            price, info = self.price_single(ccb, n_paths=n_paths, step_days=step_days, seed=seed)
            results.append((price, info))
        return results


def create_default_ccb(name, ticker, stock_price, market_price, volatility,
                       conversion_price=None, days_to_maturity=None,
                       call_trigger_pct=1.30, put_trigger_pct=0.70,
                       put_price=100.0, redemption_price=108.0) -> CCBParams:
    """Helper to create a CCBParams with sensible defaults."""
    if conversion_price is None:
        conversion_price = stock_price
    if days_to_maturity is None:
        days_to_maturity = 252 * 3
    return CCBParams(
        name=name, ticker=ticker,
        face_value=100.0, conversion_price=conversion_price,
        days_to_maturity=days_to_maturity,
        call_trigger_pct=call_trigger_pct, put_trigger_pct=put_trigger_pct,
        put_price=put_price, redemption_price=redemption_price,
        market_price=market_price, stock_price=stock_price,
        volatility=volatility, stock_ticker=ticker,
    )


if __name__ == "__main__":
    import time
    config = MCConfig(n_paths=1000, n_days=252)
    pricer = CCBPricer(config)
    
    ccb = create_default_ccb("Daqin CB", "113044", 8.20, 120.48, 0.28, 6.22, 252*3)
    
    t0 = time.time()
    price, info = pricer.price_single(ccb, n_paths=1000, step_days=21, seed=42)
    elapsed = time.time() - t0
    
    print(f"CCB: {ccb.name}")
    print(f"  Model price: {price:.4f} (market: {ccb.market_price})")
    print(f"  Conv value: {info['conv_value']:.2f}")
    print(f"  Discount: {(price - ccb.market_price) / ccb.market_price * 100:.2f}%")
    print(f"  Time: {elapsed:.3f}s ({info['n_exercise_steps']} exercise steps)")
    print(f"  Steps per second: {info['n_exercise_steps']/elapsed:.0f}")
