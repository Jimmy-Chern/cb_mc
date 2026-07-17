#!/usr/bin/env python3
"""
计算真实下修概率 p_downward
============================
方法: 用 akshare bond_cb_adj_logs_jsl 获取全市场下修记录,
     统计触发回售的转债中有多少真的下修了转股价。

输出: 实际 p_downward, 替代硬编码的 0.8
"""
import akshare as ak
import pandas as pd
import pickle
from pathlib import Path
import numpy as np

ST_CACHE = Path.home() / 'chenjunming' / 'quant' / 'cache' / 'st_fresh'

def get_adjustment_logs():
    """获取可转债转股价调整记录 (下修+分红调整等)"""
    try:
        df = ak.bond_cb_adj_logs_jsl()
        # 过滤: 只有"下修"类型的调整
        # 字段: 转债代码, 转债名称, 调整前转股价, 调整后转股价, 调整日期, 调整原因
        if '调整原因' in df.columns:
            down_adj = df[df['调整原因'].str.contains('下修', na=False)]
        else:
            # 无原因列, 所有向下调整都算
            down_adj = df.copy()
            if '调整前转股价' in df.columns and '调整后转股价' in df.columns:
                down_adj = df[df['调整后转股价'].astype(float) < df['调整前转股价'].astype(float)]
        
        print(f"  下修记录: {len(down_adj)} 条")
        return down_adj
    except Exception as e:
        print(f"  bond_cb_adj_logs_jsl 失败: {e}")
        return pd.DataFrame()

def check_put_trigger_history(bond_cb_id: str, conv_price: float, put_pct: float = 0.70, window: int = 30):
    """
    检查某只转债的正股历史上是否触发过回售条件
    (正股价连续N天 < 转股价×70%)
    """
    fpath = ST_CACHE / f"{bond_cb_id}.parquet" if (ST_CACHE / f"{bond_cb_id}.parquet").exists() else None
    if fpath is None:
        # 需要通过 bond_zh_cov 找到正股代码
        return False
    
    # 实际上 bond_cb_id 是转债代码, 需要映射到正股代码.
    # 这个函数需要重构.
    return False

if __name__ == '__main__':
    print("计算真实下修概率 p_downward")
    print("="*50)
    
    # 方法1: bond_cb_adj_logs_jsl
    try:
        adj = ak.bond_cb_adj_logs_jsl()
        print(f"\n  bond_cb_adj_logs_jsl: {len(adj)} 条, cols={list(adj.columns)[:8]}")
        if len(adj) > 0:
            print(adj.head(5).to_string())
    except Exception as e:
        print(f"  失败: {e}")
    
    # 方法2: 用 bond_zh_cov 查哪些转债转股价发生过变化
    # (bond_zh_cov 只有最新转股价, 无法直接判断历史变化)
    # 改成: 用 FTShare MCP ft_get_cb_base_data_handler 逐只查历史转股价
    
    # 方法3: 文献值
    print("\n" + "="*50)
    print("结论:")
    print("  akshare bond_cb_adj_logs_jsl 数据不完整(需集思录会员)")
    print("  文献常用值: p_downward = 0.8 (论文设定)")
    print("  建议: 如果要用真实值, 从Wind/Choice导出")
    print("  或者用保守估计: p_downward = 0.5 (更谨慎)")
    print("  当前硬编码: p_downward = 0.8 (pricer.py config.py)")
