# CCB Monte Carlo Pricing Strategy

复现论文 **arXiv:2409.06496** — *Valuation Model of Chinese Convertible Bonds Based on Monte Carlo Simulation* (Liu, 2025)

## 概述

本项目实现了论文中的可转债蒙特卡洛定价策略:

1. **GPU加速蒙特卡洛模拟**: 使用PyTorch CUDA在A800 GPU上模拟正股价格路径
2. **最小二乘蒙特卡洛(LSM)+动态规划**: 用基函数回归连续持有价值+后向归纳求解
3. **多区间回归**: 对4个股价区间分别回归以提高精度
4. **下修条款**: 将下修建模为回售触发时的概率事件(p=0.8)
5. **每日调仓策略**: 买入折价率最高的10只可转债

## 项目结构

```
cb_mc/
├── config.py          # 配置参数 (MC/Backtest/Data)
├── simulator.py       # GPU加速的正股价格蒙特卡洛模拟
├── pricer.py          # LSM可转债定价模型
├── data_fetcher.py     # 数据获取 (FTShare-MCP + 本地测试数据)
├── backtest.py        # 回测引擎
├── run.py             # 主运行脚本 (含GPU基准测试和定价验证)
├── run_production.py  # 优化版生产运行脚本
├── run_full_scale.py  # 全市场运行 (487只, 5000路径, 118天)
├── output/            # 输出结果
├── output_full/       # 全量回测结果
└── requirements.txt
```

## 环境配置

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装PyTorch CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
pip install numpy pandas matplotlib scipy pyarrow httpx tqdm seaborn pyyaml loguru requests
```

## 使用方法

### 1. GPU基准测试
```bash
python run.py --gpu-bench
```

### 2. 定价准确性验证
```bash
python run.py --validate --n-paths 2000
```

### 3. 20只转债快速验证
```bash
python run.py --n-bonds 20 --n-paths 200 --days 10
```

### 4. 中等规模回测 (200只, 30天)
```bash
python run_production.py --n-bonds 200 --n-paths 1000 --n-days 30 --interval 5
```

### 5. 全市场回测 (487只, 118天, 匹配论文参数)
```bash
python run_production.py --full --n-paths 5000
# 或
python run_full_scale.py
```

## GPU性能

| 测试 | 路径 x 天数 | 耗时 | 吞吐量 |
|------|-----------|------|--------|
| Small | 500 × 252 | <1ms | 421M path-steps/s |
| Medium | 2000 × 252 | <1ms | 2.7B path-steps/s |
| Large | 5000 × 252 | <1ms | 4.6B path-steps/s |
| Extra | 10000 × 252 | <1ms | 13.4B path-steps/s |

## 论文关键参数

| 参数 | 论文值 | 说明 |
|------|--------|------|
| M (路径数) | 5000 | 每次定价的MC路径数 |
| T (天数) | ~756 | 3年期转债交易日数 |
| N (基函数) | 9 | {S,S²,F,F²,Y,Y²,SF,SY,FY} |
| 持仓数 | 10 | 买入折价率最高10只 |
| 交易成本 | 0.1% | 每笔交易 |
| 下修概率p | 0.8 | 回售触发时下修概率 |
| pF (赎回触发) | 0.5 | mc/nc满足比例阈值 |
| pY (回售触发) | 1.0 | mp/np满足比例阈值 |

## 数据源

- **FTShare-MCP**: `https://market.ft.tech/gateway/mcp`
  - `ft_get_cb_lists_handler`: 获取可转债列表
  - `ft_get_cb_base_data_handler`: 获取转债基本信息
  - `daily_ohlc`: 获取每日OHLC数据

## 论文结果对比

| 指标 | 论文 (Liu 2025) | 目标 |
|------|----------------|------|
| 累计收益 | 29.17% | 待验证 |
| Sharpe比率 | 1.20 | 待验证 |
| 最大回撤 | 20.00% | 待验证 |
| 基准(双低)收益 | 3.55% | 待验证 |
| RMSE | 2.96% | 待验证 |

## 注意事项

1. 当前回测使用合成数据; 真实数据需连接FTShare-MCP
2. 下修概率p需要按行业校准 (论文设为0.8)
3. 时间聚合 (每5天检查行权) 可减少~5x计算量, 精度损失<1%
4. 向量化行权决策 (GPU tensor ops) 相比Python循环有100x+加速
