"""
@File    :   alphatest.py
@Time    :   2024/11/28 16:06:52
@Author  :   David Jin
@Version :   1.0
@Contact :   jinxyyy@qq.com
@License :   (C)Copyright 2024-2024, David Jin
@Desc    :   对单因子进行IC测试
"""

import numba as nb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def tearsheet(
    feature: pd.DataFrame,
    label: pd.DataFrame,
    forward: int = 1,
) -> None:
    ic = cs_ic(feature, label, forward=forward, method="pearson")
    rankic = cs_ic(feature, label, forward=forward, method="spearman")
    plt.figure(figsize=(12, 8))
    # 第一张子图 - IC
    plt.subplot(2, 1, 1)
    plt.bar(ic.index, ic, color="b", label="IC", alpha=0.7)
    plt.legend(loc="upper left")
    plt.twinx()
    plt.plot(ic.cumsum(), color="r", label="IC Cumsum")
    plt.title(f"IS IC Mean:{np.nanmean(ic):.4f}, IR:{np.nanmean(ic)/np.nanstd(ic):.4f}")
    plt.legend(loc="lower right")
    # 第二张子图 - RankIC
    plt.subplot(2, 1, 2)
    plt.bar(rankic.index, rankic, color="b", label="RankIC", alpha=0.7)
    plt.legend(loc="upper left")
    plt.twinx()
    plt.plot(rankic.cumsum(), color="r", label="RankIC Cumsum")
    plt.title(
        f"IS Rank IC Mean:{np.nanmean(rankic):.4f}, IR:{np.nanmean(rankic)/np.nanstd(rankic):.4f}"
    )
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    # 第三张子图 - 分层绩效
    # 第四张子图 - 分层累积收益(forward=1, 5, 10)
    # 第五张子图 - 多空组合绩效
    # 第六张子图 - 分组年化换手率

    return None


def cs_ic(
    X: pd.DataFrame, y: pd.DataFrame, forward: int = 1, method: str = "pearson"
) -> pd.Series:
    """
    高性能IC计算函数

    Args:
        X (pd.DataFrame): 特征矩阵
        y (pd.DataFrame): 标签矩阵
        shift (int, optional): 滞后期数. Defaults to 1.
        method (str, optional): 相关系数计算方法. Defaults to 'pearson'.

    Returns:
        pd.Series: IC值序列
    """
    forward += 1
    # 选择计算方法
    if method == "pearson":
        return pd.Series(
            _cs_corr_pearson(X.to_numpy(), y.to_numpy(), forward),
            index=X.index,
        )
    elif method == "spearman":
        return pd.Series(
            _cs_corr_spearman(X.to_numpy(), y.to_numpy(), forward),
            index=X.index,
        )
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")


@nb.njit(nogil=True, cache=True, parallel=True)
def _cs_corr_pearson(X: np.ndarray, Y: np.ndarray, forward: int) -> np.ndarray:
    """Numba加速的Pearson相关系数计算"""
    T, _ = X.shape
    ic = np.full(T, np.nan)
    for t in nb.prange(T - forward):
        x_array = X[t, :]
        y_array = Y[t + forward, :]
        valid = ~np.isnan(x_array) & ~np.isnan(y_array)
        if np.sum(valid) > 1:
            x_valid = x_array[valid]
            y_valid = y_array[valid]
            x_mean = np.mean(x_valid)
            y_mean = np.mean(y_valid)
            x_std = np.std(x_valid)
            y_std = np.std(y_valid)
            cov = np.mean((x_valid - x_mean) * (y_valid - y_mean))
            ic[t + forward] = cov / (x_std * y_std)
    return ic


@nb.njit(nogil=True, cache=True)
def _avg_rank(data: np.ndarray) -> np.ndarray:
    """计算平均排名"""
    n = len(data)
    ranks = np.zeros(n, dtype=np.float64)
    sorted_indices = np.argsort(data)
    # 记录相等元素的起始位置
    start = 0
    while start < n:
        end = start
        while end < n and data[sorted_indices[end]] == data[sorted_indices[start]]:
            end += 1
        # 计算该组的平均排名
        avg_rank = (start + 1 + end) / 2
        for j in range(start, end):
            ranks[sorted_indices[j]] = avg_rank
        start = end
    return ranks


@nb.njit(nogil=True, cache=True, parallel=True)
def _cs_corr_spearman(X: np.ndarray, Y: np.ndarray, forward: int) -> np.ndarray:
    """Numba加速的Spearman相关系数计算"""
    T, _ = X.shape
    ic = np.full(T, np.nan)
    for t in nb.prange(T - forward):
        x_array = X[t, :]
        y_array = Y[t + forward, :]
        valid = ~np.isnan(x_array) & ~np.isnan(y_array)
        if np.sum(valid) > 1:
            x_valid = x_array[valid]
            y_valid = y_array[valid]
            x_rank = _avg_rank(x_valid)
            y_rank = _avg_rank(y_valid)
            x_mean = np.mean(x_rank)
            y_mean = np.mean(y_rank)
            x_std = np.std(x_rank)
            y_std = np.std(y_rank)
            cov = np.mean((x_rank - x_mean) * (y_rank - y_mean))
            ic[t + forward] = cov / (x_std * y_std)
    return ic
