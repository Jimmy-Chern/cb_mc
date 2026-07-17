#!/usr/bin/env python3
"""Quick end-to-end validation of the CCB MC pricing pipeline."""
import sys, time
sys.path.insert(0, '.')

import torch
import numpy as np
import pandas as pd
from config import MCConfig, BacktestConfig, DataConfig
from pricer import CCBPricer, CCBParams, create_default_ccb
from simulator import StockPathSimulator
from data_fetcher import DataManager
from backtest import CCBBacktestEngine

def main():
    print("=" * 50)
    print("E2E Validation: CCB MC Pricing Pipeline")
    print("=" * 50)
    
    # 1. Test GPU
    print(f"\n[1] GPU check: {torch.cuda.get_device_name(0)}")
    
    # 2. Simulator
    print("\n[2] Testing stock path simulator...")
    config = MCConfig(n_paths=500, n_days=252, use_gpu=True)
    sim = StockPathSimulator(config)
    t0 = time.time()
    paths = sim.simulate_paths(100.0, 0.3, 0.03, seed=42)
    gpu_time = time.time() - t0
    print(f"   500 paths x 252 days: {gpu_time:.4f}s, shape={paths.shape}")
    
    # 3. Single CCB pricing
    print("\n[3] Testing single CCB pricing...")
    pricer = CCBPricer(config)
    ccb = create_default_ccb("TestCB", "113044", 8.20, 120.48, 0.28, 6.22, 252*3)
    
    t0 = time.time()
    price, info = pricer.price_single(ccb, n_paths=200, seed=42)
    p_time = time.time() - t0
    conv_val = 100/6.22 * 8.20
    print(f"   Model price: {price:.2f} (conv value: {conv_val:.2f})")
    print(f"   Pricing time: {p_time:.3f}s")
    print(f"   Info: {info}")
    
    # 4. Batch pricing (5 bonds)
    print("\n[4] Testing batch pricing (5 bonds)...")
    bonds = [create_default_ccb(f"CB{i}", f"T{i:06d}", 
                                8+np.random.randn()*2, 100+np.random.randn()*10, 
                                0.25+np.random.rand()*0.2, 8+np.random.randn())
             for i in range(5)]
    
    t0 = time.time()
    results = []
    for b in bonds:
        p, info = pricer.price_single(b, n_paths=100, seed=42)
        results.append((p, info))
        print(f"   {b.name}: model={p:.2f}, market={b.market_price:.2f}, "
              f"discount={(p-b.market_price)/b.market_price*100:.2f}%")
    batch_time = time.time() - t0
    print(f"   5 bonds total: {batch_time:.3f}s ({batch_time/5:.3f}s each)")
    
    # 5. Full pipeline test (5 bonds, 3 days)
    print("\n[5] Testing mini backtest (5 bonds, 3 days)...")
    data_config = DataConfig(test_n_bonds=5)
    data_mgr = DataManager(data_config)
    bond_df = data_mgr.fetch_bond_universe(use_mcp=False)
    
    stock_hist = {}
    for _, row in bond_df.iterrows():
        tkr = row["ticker"]
        hist = data_mgr._generate_synthetic_prices(
            "2023-01-01", "2023-07-01",
            S0=float(row["stock_price"]),
            sigma=float(row["volatility"])
        )
        stock_hist[tkr] = hist
    
    bt_config = BacktestConfig(n_positions=3)
    engine = CCBBacktestEngine(
        mc_config=MCConfig(n_paths=100),
        bt_config=bt_config,
        data_config=data_config,
        output_dir="./output",
    )
    
    trading_days = pd.bdate_range("2023-02-18", "2023-02-22")[:3]
    
    t0 = time.time()
    result = engine.run(
        bond_universe=bond_df,
        stock_price_history=stock_hist,
        trading_days=trading_days,
        n_positions=3,
        n_paths=50,  # Small for speed
    )
    bt_time = time.time() - t0
    
    print(f"   Backtest time: {bt_time:.3f}s")
    print(f"   Portfolio: {len(result.returns)} days")
    print(f"   Cum return: {result.cumulative_return*100:.2f}%")
    print(f"   Sharpe: {result.sharpe_ratio:.2f}")
    
    # 6. Perf estimate
    print(f"\n[6] Performance Estimate:")
    print(f"   GPU throughput: {500*252/gpu_time:.0f} path-steps/s")
    print(f"   Single pricing (200 paths): {p_time:.3f}s")
    print(f"   Est. 20 bonds x 10 days x 200 paths: {20*10*p_time:.1f}s")
    print(f"   Est. 487 bonds x 118 days x 5000 paths: {487*118*p_time*5000/200:.1f}s")
    print(f"   Est. w/ A800 batch (50% gain): {487*118*p_time*5000/200*0.5:.1f}s")
    
    print("\n✓ Pipeline validated!")
    return True

if __name__ == "__main__":
    main()
