#!/usr/bin/env python3
"""
Main run script: Reproduce CCB Monte Carlo Pricing Strategy from arxiv:2409.06496.

Workflow:
1. Fetch convertible bond data (MCP API or local test data)
2. Price each CCB using Monte Carlo simulation + dynamic programming
3. Run backtest with daily rebalance
4. Generate performance report and plots

Usage:
    python run.py                    # Run with 20 test bonds
    python run.py --full             # Run full market
    python run.py --n-bonds 50       # Run with N bonds
    python run.py --n-paths 2000     # Use fewer MC paths for speed
    python run.py --skip-pricing     # Skip pricing, use pre-computed
    python run.py --days 60          # Simulate N trading days
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MCConfig, BacktestConfig, DataConfig
from pricer import CCBPricer, CCBParams, create_default_ccb
from simulator import StockPathSimulator, historical_volatility
from data_fetcher import DataManager, LocalDataProvider
from backtest import CCBBacktestEngine, BacktestResult


def setup_logging(output_dir: str):
    """Configure logging."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        log_dir / "run_{time}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )


def test_gpu_performance(config: MCConfig, output_dir: str):
    """Benchmark GPU vs CPU performance for MC simulation."""
    logger.info("=== GPU Performance Test ===")
    
    sim = StockPathSimulator(config)
    
    # Test parameters
    test_configs = [
        (500, 252),
        (1000, 252),
        (2000, 252),
        (5000, 252),
        (10000, 252),
    ]
    
    results = []
    device_name = "GPU" if sim.device.type == "cuda" else "CPU"
    
    for n_paths, n_days in test_configs:
        # Warmup
        _ = sim.simulate_paths(100.0, 0.3, 0.03, n_days=n_days, n_paths=n_paths)
        
        # Benchmark
        torch.cuda.synchronize() if sim.device.type == "cuda" else None
        t0 = time.time()
        for _ in range(3):
            paths = sim.simulate_paths(100.0, 0.3, 0.03, n_days=n_days, n_paths=n_paths)
        torch.cuda.synchronize() if sim.device.type == "cuda" else None
        elapsed = time.time() - t0
        
        paths_per_sec = (3 * n_paths * n_days) / elapsed
        mem_gb = paths.element_size() * paths.nelement() / (1024**3)
        
        results.append({
            "n_paths": n_paths,
            "n_days": n_days,
            "time_3_runs": f"{elapsed:.3f}s",
            "paths_per_sec": f"{paths_per_sec:.0f}",
            "memory_gb": f"{mem_gb:.3f}",
        })
        logger.info(f"  {n_paths}x{n_days}: {elapsed:.3f}s ({paths_per_sec:.0f} path-steps/s, {mem_gb:.2f} GB)")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(Path(output_dir) / "gpu_benchmark.csv", index=False)
    
    return df


def validate_pricing_accuracy(pricer: CCBPricer, output_dir: str):
    """Validate pricing accuracy against known test cases."""
    logger.info("=== Pricing Accuracy Validation ===")
    
    # Test cases from paper Table 4 (Daqin CB example)
    test_cases = [
        {
            "name": "Daqin CB (High Premium)",
            "params": create_default_ccb("Daqin", "113044", 8.20, 120.48, 0.28, 6.22),
            "expected_range": (115, 130),
        },
        {
            "name": "At-the-money",
            "params": create_default_ccb("ATM", "TEST01", 10.0, 100.0, 0.30, 10.0),
            "expected_range": (95, 115),
        },
        {
            "name": "Deep in-the-money",
            "params": create_default_ccb("DITM", "TEST02", 20.0, 190.0, 0.30, 10.0),
            "expected_range": (175, 210),
        },
        {
            "name": "Deep out-of-the-money",
            "params": create_default_ccb("DOTM", "TEST03", 3.0, 80.0, 0.40, 10.0),
            "expected_range": (75, 95),
        },
    ]
    
    results = []
    for tc in test_cases:
        ccb = tc["params"]
        price, info = pricer.price_single(ccb, n_paths=2000)
        in_range = tc["expected_range"][0] <= price <= tc["expected_range"][1]
        
        conv_value = ccb.face_value / ccb.conversion_price * ccb.stock_price
        discount = (price - ccb.market_price) / ccb.market_price * 100 if ccb.market_price > 0 else 0
        
        status = "✓" if in_range else "✗"
        logger.info(
            f"  {status} {tc['name']}: "
            f"model={price:.2f}, market={ccb.market_price:.2f}, "
            f"conv={conv_value:.2f}, disc={discount:.2f}%"
        )
        
        results.append({
            "name": tc["name"],
            "model_price": price,
            "market_price": ccb.market_price,
            "conv_value": conv_value,
            "discount_pct": discount,
            "in_range": in_range,
        })
    
    df = pd.DataFrame(results)
    df.to_csv(Path(output_dir) / "pricing_validation.csv", index=False)
    
    n_pass = sum(1 for r in results if r["in_range"])
    logger.info(f"  Validation: {n_pass}/{len(results)} test cases in expected range")
    
    return df


def run_backtest_20_bonds(output_dir: str, n_bonds: int = 20, n_paths: int = 500, 
                          n_days: int = 10, n_positions: int = 10):
    """Run the full backtest with configurable parameters."""
    logger.info("=" * 60)
    logger.info(f"RUNNING {n_bonds}-BOND BACKTEST ({n_days} days, {n_paths} paths)")
    logger.info("=" * 60)
    
    # Configuration
    mc_config = MCConfig(
        n_paths=n_paths,
        n_days=252,
        rfr=0.03,
        transaction_cost=0.001,
        use_gpu=True,
        dtype="float32",
    )
    bt_config = BacktestConfig(
        n_positions=n_positions,
        start_date="2023-02-18",
        end_date="2023-07-17",
    )
    data_config = DataConfig(
        test_n_bonds=n_bonds,
    )
    
    # Get data
    data_mgr = DataManager(data_config)
    bond_universe = data_mgr.fetch_bond_universe(use_mcp=False)
    logger.info(f"Bond universe: {len(bond_universe)} bonds")
    
    # Generate synthetic price history for each stock
    stock_price_history = {}
    for _, row in bond_universe.iterrows():
        ticker = row.get("ticker", "unknown")
        hist = data_mgr._generate_synthetic_prices(
            bt_config.start_date, bt_config.end_date,
            S0=float(row.get("stock_price", 100)),
            sigma=float(row.get("volatility", 0.3)),
        )
        stock_price_history[ticker] = hist
    
    # Initialize engine
    engine = CCBBacktestEngine(
        mc_config=mc_config,
        bt_config=bt_config,
        data_config=data_config,
        output_dir=output_dir,
    )
    
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Generate trading days
    trading_days = pd.bdate_range(bt_config.start_date, bt_config.end_date)
    actual_days = min(n_days, len(trading_days))
    logger.info(f"Trading days available: {len(trading_days)}, using: {actual_days}")
    
    # Run backtest
    t0 = time.time()
    result = engine.run(
        bond_universe=bond_universe,
        stock_price_history=stock_price_history,
        trading_days=trading_days[:actual_days],
        n_positions=n_positions,
        n_paths=n_paths,
    )
    elapsed = time.time() - t0
    logger.info(f"Backtest completed in {elapsed:.1f}s")
    
    # Run benchmark
    benchmark = engine.run_benchmark(bond_universe)
    result.benchmark_returns = benchmark.returns
    result.benchmark_cum_return = benchmark.cumulative_return
    
    # Generate report
    report = generate_report(result, benchmark, output_dir)
    
    # Generate plots
    engine.plot_results(result, benchmark)
    engine.plot_pricing_accuracy(result)
    
    logger.info("\n" + result.summary())
    
    return result


def generate_report(result: BacktestResult, benchmark: BacktestResult, output_dir: str) -> str:
    """Generate a comprehensive markdown backtest report."""
    report_lines = [
        "# CCB Monte Carlo Pricing Strategy - Backtest Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Methodology",
        "",
        "Based on Liu (2025) - *Valuation Model of Chinese Convertible Bonds Based on Monte Carlo Simulation* (arXiv:2409.06496)",
        "",
        "### Pricing Model",
        "- Least Squares Monte Carlo (LSM) with dynamic programming",
        "- 500-5000 Monte Carlo paths per bond per day",
        "- Multi-regression across 4 stock price intervals",
        "- Basis functions: {S, S², F, F², Y, Y², SF, SY, FY}",
        "- Downward adjustment modeled as probabilistic event (p=0.8)",
        "",
        "### Trading Strategy",
        "- Long top 10 most undervalued CCBs (by model discount)",
        "- Daily rebalance with 0.1% transaction cost",
        "- Equal-weight portfolio",
        "",
        "### Benchmark",
        "- Double Low strategy (low price + low conversion premium)",
        "",
        "---",
        "",
        "## Results",
        "",
        "### Performance Metrics",
        "",
        "| Metric | LSM Strategy | Double Low |",
        "|--------|-------------|------------|",
        f"| Cumulative Return | {result.cumulative_return*100:.2f}% | {benchmark.cumulative_return*100:.2f}% |",
        f"| Annualized Return | {result.annualized_return*100:.2f}% | {benchmark.annualized_return*100:.2f}% |",
        f"| Annualized Vol | {result.annualized_vol*100:.2f}% | {benchmark.annualized_vol*100:.2f}% |",
        f"| Sharpe Ratio | {result.sharpe_ratio:.2f} | {benchmark.sharpe_ratio:.2f} |",
        f"| Max Drawdown | {result.max_drawdown*100:.2f}% | {benchmark.max_drawdown*100:.2f}% |",
        f"| Win Rate (daily) | {result.win_rate*100:.1f}% | {benchmark.win_rate*100:.1f}% |",
        "",
        "### Comparison with Paper Results",
        "",
        "| Factor | Paper (Liu 2025) | This Implementation |",
        "|--------|-----------------|-------------------|",
        "| Cumulative Return | 29.17% | TBD |",
        "| Sharpe Ratio | 1.20 | TBD |",
        "| Max Drawdown | 20.00% | TBD |",
        "| Benchmark Return | 3.55% | TBD |",
        "| Benchmark Sharpe | 0.38 | TBD |",
        "| Benchmark Max DD | 23.89% | TBD |",
        "",
        "---",
        "",
        "## Files",
        "",
        f"- Backtest plot: `{output_dir}/backtest_results.png`",
        f"- Pricing accuracy: `{output_dir}/pricing_accuracy.png`",
        f"- GPU benchmark: `{output_dir}/gpu_benchmark.csv`",
        f"- Pricing validation: `{output_dir}/pricing_validation.csv`",
        "",
        "## Notes",
        "",
        "- The paper uses 5000 paths with full 10 bonds over 118 trading days",
        "- GPU acceleration on A800 80GB enables much faster batch pricing",
        "- Current run uses reduced parameters for validation speed",
    ]
    
    report_path = Path(output_dir) / "backtest_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Report saved to {report_path}")
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(
        description="CCB Monte Carlo Pricing Backtest (Liu 2025, arxiv:2409.06496)"
    )
    parser.add_argument("--n-bonds", type=int, default=20, help="Number of bonds to test")
    parser.add_argument("--n-paths", type=int, default=2000, help="MC paths per pricing")
    parser.add_argument("--n-positions", type=int, default=10, help="Portfolio positions")
    parser.add_argument("--days", type=int, default=10, help="Trading days to simulate")
    parser.add_argument("--full", action="store_true", help="Run full market simulation")
    parser.add_argument("--skip-pricing", action="store_true", help="Skip pricing (use cached)")
    parser.add_argument("--gpu-bench", action="store_true", help="Run GPU benchmark only")
    parser.add_argument("--validate", action="store_true", help="Run pricing validation only")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(str(output_dir))
    
    logger.info("=" * 60)
    logger.info("CCB Monte Carlo Pricing System")
    logger.info(f"Paper: arxiv:2409.06496 (Liu 2025)")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("No GPU detected, using CPU (will be slow)")
    
    # GPU benchmark
    if args.gpu_bench:
        config = MCConfig(n_paths=args.n_paths, use_gpu=True)
        test_gpu_performance(config, str(output_dir))
        return
    
    # Pricing validation
    if args.validate:
        config = MCConfig(n_paths=args.n_paths, use_gpu=True)
        pricer = CCBPricer(config)
        validate_pricing_accuracy(pricer, str(output_dir))
        return
    
    # Run backtest
    logger.info(f"\nStarting backtest with {args.n_bonds} bonds, {args.days} days, {args.n_paths} paths...")
    result = run_backtest_20_bonds(
        str(output_dir),
        n_bonds=args.n_bonds,
        n_paths=args.n_paths,
        n_days=args.days,
        n_positions=args.n_positions,
    )
    
    logger.success("Backtest complete!")
    logger.info(f"Results saved to {output_dir}/")
    for f in output_dir.glob("*"):
        logger.info(f"  {f.name}")


if __name__ == "__main__":
    main()
