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

from AlphaEngine.operators import numba as op


def ic(
    X: pd.DataFrame,
    y: pd.DataFrame,
    forward: int = 1,
    method: str = "pearson",
    aggsize: str = "Y",
    period: int = 252,
) -> pd.Series:
    """
    时序IC计算

    Args:
        X (pd.DataFrame): 特征矩阵
        y (pd.DataFrame): 标签矩阵
        shift (int, optional): 滞后期数. Defaults to 1.
        method (str, optional): 相关系数计算方法. Defaults to 'pearson'.
        aggsize (str, optional): 聚合周期. Defaults to 'Y'.
        period (int, optional): 滚动标准化窗口大小. Defaults to 252.

    Returns:
        pd.Series: IC值表格
    """
    forward += 1
    X = X.copy()
    y = y.copy()
    X = _preprocess(X, forward, period)
    # 选择聚合周期
    if aggsize == "Y":
        X["group"] = X.index.year
        y["group"] = y.index.year
    elif aggsize == "M":
        X["group"] = X.index.month
        y["group"] = y.index.month
    elif aggsize == "D":
        X["group"] = X.index.day
        y["group"] = y.index.day
    else:
        raise ValueError("aggsize must be 'Y', 'M' or 'D'")
    # 计算IC
    if method == "pearson":
        ic = pd.DataFrame(
            _ts_corr_pearson(X.to_numpy(), y.to_numpy()),
            index=X["group"].unique(),
            columns=X.columns[:-1],
        )
    elif method == "spearman":
        ic = pd.DataFrame(
            _ts_corr_spearman(X.to_numpy(), y.to_numpy()),
            index=X["group"].unique(),
            columns=X.columns[:-1],
        )
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
    return ic


def _preprocess(X: pd.DataFrame, forward: int, p: int) -> pd.DataFrame:
    """数据预处理"""
    return op.ts_delay(X, forward)


@nb.njit(nogil=True, cache=True, parallel=True)
def _ts_corr_pearson(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    group = np.unique(X[:, -1])
    N = X.shape[1] - 1
    T = len(group)
    ic = np.full((T, N), np.nan)
    for col in nb.prange(N):
        for t in range(T):
            x_array = X[X[:, -1] == group[t], col]
            y_array = Y[Y[:, -1] == group[t], col]
            valid = np.isfinite(x_array) & np.isfinite(y_array)
            if np.sum(valid) > 2:
                x_valid = x_array[valid]
                y_valid = y_array[valid]
                x_mean = np.mean(x_valid)
                y_mean = np.mean(y_valid)
                x_std = np.std(x_valid)
                y_std = np.std(y_valid)
                cov = np.mean((x_valid - x_mean) * (y_valid - y_mean))
                ic[t, col] = cov / (x_std * y_std)
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
def _ts_corr_spearman(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    group = np.unique(X[:, -1])
    N = X.shape[1] - 1
    T = len(group)
    ic = np.full((T, N), np.nan)
    for col in nb.prange(N):
        for t in range(T):
            x_array = X[X[:, -1] == group[t], col]
            y_array = Y[Y[:, -1] == group[t], col]
            valid = np.isfinite(x_array) & np.isfinite(y_array)
            if np.sum(valid) > 2:
                x_valid = x_array[valid]
                y_valid = y_array[valid]
                x_rank = _avg_rank(x_valid)
                y_rank = _avg_rank(y_valid)
                x_mean = np.mean(x_rank)
                y_mean = np.mean(y_rank)
                x_std = np.std(x_rank)
                y_std = np.std(y_rank)
                cov = np.mean((x_rank - x_mean) * (y_rank - y_mean))
                ic[t, col] = cov / (x_std * y_std)
    return ic
