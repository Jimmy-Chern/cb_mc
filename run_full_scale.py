#!/usr/bin/env python3
"""
Full-scale CCB Monte Carlo Pricing Backtest - arxiv:2409.06496
Run with 487 bonds, 5000 paths, 118 trading days (matching the paper).
Expects ~2-3 hours on A800 GPU.

Usage: python run_full_scale.py
"""

import subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).parent

def main():
    cmd = [
        sys.executable, str(ROOT / "run_production.py"),
        "--n-bonds", "487",
        "--n-paths", "5000",
        "--n-days", "118",
        "--interval", "5",
        "--output", str(ROOT / "output_full"),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Estimated time: ~2 hours on A800")
    print(f"Paper: Liu (2025) - arxiv:2409.06496")
    print("-" * 60)
    
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
