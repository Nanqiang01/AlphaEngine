"""
@File    :   numba.py
@Time    :   2024/11/27 23:34:03
@Author  :   David Jin
@Version :   1.0
@Contact :   jinxyyy@qq.com
@License :   (C)Copyright 2024-2024, David Jin
@Desc    :   定义了一系列基于Numpy和Numba的算子函数，用于实现元素运算、时序运算、截面运算和逻辑运算。
"""

import numpy as np
import pandas as pd

from AlphaEngine.operators import _numba as _nb


# 输入检查
def _check_input(X: pd.DataFrame) -> None:
    """检查输入类型是否为DataFrame"""
    if isinstance(X, pd.DataFrame):
        pass
    else:
        raise TypeError("输入类型错误！")


# 二维输入检查
def _check_shape(X: pd.DataFrame, Y: pd.DataFrame) -> None:
    """检查矩阵大小是否一致"""
    if isinstance(X, pd.DataFrame) and isinstance(Y, pd.DataFrame):
        if np.any(X.index != Y.index) or np.any(X.columns != Y.columns):
            raise ValueError("X和Y的index或columns不一致！")
    else:
        raise TypeError("X和Y的类型不一致！")


########## 元素算子 ##########


def add(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """
    x + y
    """
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._add(X.to_numpy(), Y.to_numpy()), index=X.index, columns=X.columns
    )


def sub(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """
    x - y
    """
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._sub(X.to_numpy(), Y.to_numpy()), index=X.index, columns=X.columns
    )


def mul(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """
    x * y
    """
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._mul(X.to_numpy(), Y.to_numpy()), index=X.index, columns=X.columns
    )


def max(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """
    max(x, y)
    """
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._max(X.to_numpy(), Y.to_numpy()), index=X.index, columns=X.columns
    )


def min(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """
    min(x, y)
    """
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._min(X.to_numpy(), Y.to_numpy()), index=X.index, columns=X.columns
    )


def div(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """
    x / y
    """
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._div(X.to_numpy(), Y.to_numpy()), index=X.index, columns=X.columns
    )


def ceiling(X: pd.DataFrame) -> pd.DataFrame:
    """
    x向上取整
    """
    _check_input(X)
    return pd.DataFrame(_nb._ceiling(X.to_numpy()), index=X.index, columns=X.columns)


def floor(X: pd.DataFrame) -> pd.DataFrame:
    """
    x向下取整
    """
    _check_input(X)
    return pd.DataFrame(_nb._floor(X.to_numpy()), index=X.index, columns=X.columns)


def round(X: pd.DataFrame, n: int = 0) -> pd.DataFrame:
    """
    x四舍五入
    """
    _check_input(X)
    return pd.DataFrame(_nb._round(X.to_numpy(), n), index=X.index, columns=X.columns)


def fraction(X: pd.DataFrame) -> pd.DataFrame:
    """
    x的小数部分
    """
    _check_input(X)
    return pd.DataFrame(_nb._fraction(X.to_numpy()), index=X.index, columns=X.columns)


def log(X: pd.DataFrame) -> pd.DataFrame:
    """
    log(x)
    """
    _check_input(X)
    return pd.DataFrame(_nb._log(X.to_numpy()), index=X.index, columns=X.columns)


def exp(X: pd.DataFrame) -> pd.DataFrame:
    """
    e^x
    """
    _check_input(X)
    return pd.DataFrame(_nb._exp(X.to_numpy()), index=X.index, columns=X.columns)


def sqrt(X: pd.DataFrame) -> pd.DataFrame:
    """
    sqrt(x)
    """
    _check_input(X)
    return pd.DataFrame(_nb._sqrt(X.to_numpy()), index=X.index, columns=X.columns)


def pow(X: pd.DataFrame, a: int = 2) -> pd.DataFrame:
    """
    x ^ a
    """
    _check_input(X)
    return pd.DataFrame(_nb._pow(X.to_numpy(), a), index=X.index, columns=X.columns)


def abs(X: pd.DataFrame) -> pd.DataFrame:
    """
    |x|
    """
    _check_input(X)
    return pd.DataFrame(_nb._abs(X.to_numpy()), index=X.index, columns=X.columns)


def sign(X: pd.DataFrame) -> pd.DataFrame:
    """
    if input = NaN; return NaN
    else if input > 0, return 1
    else if input < 0, return -1
    else if input = 0, return 0
    """
    _check_input(X)
    return pd.DataFrame(_nb._sign(X.to_numpy()), index=X.index, columns=X.columns)


def neg(X: pd.DataFrame) -> pd.DataFrame:
    """
    -x
    """
    _check_input(X)
    return pd.DataFrame(_nb._neg(X.to_numpy()), index=X.index, columns=X.columns)


def inv(X: pd.DataFrame) -> pd.DataFrame:
    """
    1 / x
    """
    _check_input(X)
    return pd.DataFrame(_nb._inv(X.to_numpy()), index=X.index, columns=X.columns)


def sigmoid(X: pd.DataFrame) -> pd.DataFrame:
    """
    1 / (1 + exp(-x))
    """
    _check_input(X)
    return pd.DataFrame(_nb._sigmoid(X.to_numpy()), index=X.index, columns=X.columns)


def signed_power(X: pd.DataFrame, a: int) -> pd.DataFrame:
    """
    sign(x) * (abs(x) ^ y)
    x的y次方，最终结果保留x的符号
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._signed_power(X.to_numpy(), a), index=X.index, columns=X.columns
    )


def s_log_1p(X: pd.DataFrame) -> pd.DataFrame:
    """
    sign(x) * log(1 + abs(x))
    """
    _check_input(X)
    return pd.DataFrame(_nb._s_log_1p(X.to_numpy()), index=X.index, columns=X.columns)


def cos(X: pd.DataFrame) -> pd.DataFrame:
    """
    cos(x)
    """
    _check_input(X)
    return pd.DataFrame(_nb._cos(X.to_numpy()), index=X.index, columns=X.columns)


def sin(X: pd.DataFrame) -> pd.DataFrame:
    """
    sin(x)
    """
    _check_input(X)
    return pd.DataFrame(_nb._sin(X.to_numpy()), index=X.index, columns=X.columns)


def tan(X: pd.DataFrame) -> pd.DataFrame:
    """
    tan(x)
    """
    _check_input(X)
    return pd.DataFrame(_nb._tan(X.to_numpy()), index=X.index, columns=X.columns)


def nan_mask(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """
    如果x对应的y小于0，则替换为NAN
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._nan_mask(X.to_numpy(), Y.to_numpy()), index=X.index, columns=X.columns
    )


def nan_out(X: pd.DataFrame, lower: int = 0, upper: int = 0) -> pd.DataFrame:
    """
    如果x小于lower或大于upper，则替换为NAN
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._nan_out(X.to_numpy(), lower, upper), index=X.index, columns=X.columns
    )


def to_nan(X: pd.DataFrame, value: int = 0, reverse: bool = False) -> pd.DataFrame:
    """
    如果reverse为False，则将value替换为NAN，否则将value替换为NAN
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._to_nan(X.to_numpy(), value, reverse), index=X.index, columns=X.columns
    )


def purify(X: pd.DataFrame) -> pd.DataFrame:
    """
    将包含多个类型的分组字段转换为数量较少的可用类型，从而提高分组字段的计算效率。
    """
    _check_input(X)
    return pd.DataFrame(_nb._purify(X.to_numpy()), index=X.index, columns=X.columns)


########## 逻辑算子 ##########


def and_(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """
    x & y
    """
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._and(X.to_numpy(), Y.to_numpy()), index=X.index, columns=X.columns
    )


def or_(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """
    x | y
    """
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._or(X.to_numpy(), Y.to_numpy()), index=X.index, columns=X.columns
    )


def equal(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """
    x == y
    """
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._equal(X.to_numpy(), Y.to_numpy()), index=X.index, columns=X.columns
    )


def negate(X: pd.DataFrame) -> pd.DataFrame:
    """
    ~x
    """
    _check_input(X)
    return pd.DataFrame(_nb._negate(X.to_numpy()), index=X.index, columns=X.columns)


def less(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """
    x < y
    """
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._less(X.to_numpy(), Y.to_numpy()), index=X.index, columns=X.columns
    )


def if_then_else(
    condition: pd.DataFrame, out1: pd.DataFrame, out2: pd.DataFrame
) -> pd.DataFrame:
    """条件判断"""
    _check_input(condition)
    _check_input(out1)
    _check_input(out2)
    _check_shape(out1, out2)
    return pd.DataFrame(
        _nb._if_then_else(condition.to_numpy(), out1.to_numpy(), out2.to_numpy()),
        index=condition.index,
        columns=condition.columns,
    )


def is_not_nan(X: pd.DataFrame) -> pd.DataFrame:
    """判断是否为非空"""
    _check_input(X)
    return pd.DataFrame(_nb._is_not_nan(X.to_numpy()), index=X.index, columns=X.columns)


def is_nan(X: pd.DataFrame) -> pd.DataFrame:
    """判断是否为空"""
    _check_input(X)
    return pd.DataFrame(_nb._is_nan(X.to_numpy()), index=X.index, columns=X.columns)


def is_finite(X: pd.DataFrame) -> pd.DataFrame:
    """判断是否为有限值"""
    _check_input(X)
    return pd.DataFrame(_nb._is_finite(X.to_numpy()), index=X.index, columns=X.columns)


def is_not_finite(X: pd.DataFrame) -> pd.DataFrame:
    """判断是否为无限值"""
    _check_input(X)
    return pd.DataFrame(
        _nb._is_not_finite(X.to_numpy()), index=X.index, columns=X.columns
    )


########## 时序算子 ##########


def time_from_last_change(X: pd.DataFrame) -> pd.DataFrame:
    """距离上次变化的时间"""
    _check_input(X)
    return pd.DataFrame(
        _nb._time_from_last_change(X.to_numpy()), index=X.index, columns=X.columns
    )


def ts_delay(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    滞后p期
    """
    _check_input(X)
    # return pd.DataFrame(
    #     _nb._ts_delay(X.to_numpy(), p), index=X.index, columns=X.columns
    # )
    return X.shift(p)


def ts_delta(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    一阶差分
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_delta(X.to_numpy(), p), index=X.index, columns=X.columns
    )


def ts_delta_pct(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    一阶差分（百分比）
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_delta_pct(X.to_numpy(), p), index=X.index, columns=X.columns
    )


def ts_log_diff(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    log(x) - log(p期前的x)
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_log_diff(X.to_numpy(), p), index=X.index, columns=X.columns
    )


def ts_av_diff(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    平均差分
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_av_diff(X.to_numpy(), p), index=X.index, columns=X.columns
    )


def ts_sum(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    过去p期求和
    """
    _check_input(X)
    return X.rolling(p).sum(engine="numba")


def ts_cumsum(X: pd.DataFrame) -> pd.DataFrame:
    """
    累计求和
    """
    _check_input(X)
    # return pd.DataFrame(_nb._ts_cumsum(X.to_numpy()), index=X.index, columns=X.columns)
    return X.cumsum()


def ts_mean(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    过去p期均值
    """
    _check_input(X)
    return X.rolling(p).mean(engine="numba")


def ts_std(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    过去p期标准差
    """
    _check_input(X)
    return X.rolling(p).std(engine="numba")


def ts_cov(X: pd.DataFrame, Y: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    过去p期协方差
    """
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._ts_cov(X.to_numpy(), Y.to_numpy(), p),
        index=X.index,
        columns=X.columns,
    )


def ts_corr(X: pd.DataFrame, Y: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    过去p期相关系数
    """
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._ts_corr(X.to_numpy(), Y.to_numpy(), p),
        index=X.index,
        columns=X.columns,
    )


def ts_triple_corr(
    X: pd.DataFrame, Y: pd.DataFrame, Z: pd.DataFrame, p: int
) -> pd.DataFrame:
    """
    过去p期三元相关系数
    """
    _check_input(X)
    _check_input(Y)
    _check_input(Z)
    _check_shape(X, Y)
    _check_shape(X, Z)
    return pd.DataFrame(
        _nb._ts_triple_corr(X.to_numpy(), Y.to_numpy(), Z.to_numpy(), p),
        index=X.index,
        columns=X.columns,
    )


def ts_partial_corr(
    X: pd.DataFrame, Y: pd.DataFrame, Z: pd.DataFrame, p: int
) -> pd.DataFrame:
    """
    过去p期偏相关系数
    """
    _check_input(X)
    _check_input(Y)
    _check_input(Z)
    _check_shape(X, Y)
    _check_shape(X, Z)
    return pd.DataFrame(
        _nb._ts_partial_corr(X.to_numpy(), Y.to_numpy(), Z.to_numpy(), p),
        index=X.index,
        columns=X.columns,
    )


def ts_rank(
    X: pd.DataFrame, p: int, method: str | None = "average", pct: bool | None = True
) -> pd.DataFrame:
    """
    过去p期排序，支持min、max、average、first、dense五种方法
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_rank(X.to_numpy(), p, method, pct), index=X.index, columns=X.columns
    )


def ts_percentage(X: pd.DataFrame, p: int, percentage: float = 0.5) -> pd.DataFrame:
    """
    过去p期大于等于percentage位数的值
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_percentage(X.to_numpy(), p, percentage),
        index=X.index,
        columns=X.columns,
    )


def ts_max(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    过去p期最大值
    """
    _check_input(X)
    return X.rolling(p).max(engine="numba")


def ts_max_diff(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    过去p期最大值差
    """
    _check_input(X)
    return X - ts_max(X, p)


def ts_min(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    过去p期最小值
    """
    _check_input(X)
    return X.rolling(p).min(engine="numba")


def ts_min_diff(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    过去p期最小值差
    """
    _check_input(X)
    return X - ts_min(X, p)


def ts_min_max_cps(X: pd.DataFrame, p: int, f: float = 2) -> pd.DataFrame:
    """
    ts_max(X, p) + ts_min(X, p) - f * X
    """
    _check_input(X)
    return ts_max(X, p) + ts_min(X, p) - f * X


def ts_min_max_diff(X: pd.DataFrame, p: int, f: float = 0.5) -> pd.DataFrame:
    """
    X - f * (ts_max(X, p) + ts_min(X, p))
    """
    _check_input(X)
    return X - f * (ts_max(X, p) + ts_min(X, p))


def ts_cummax(X: pd.DataFrame) -> pd.DataFrame:
    """累计最大值"""
    _check_input(X)
    return X.cummax()


def ts_cummin(X: pd.DataFrame) -> pd.DataFrame:
    """累计最小值"""
    _check_input(X)
    return X.cummin()


def ts_argmax(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """过去p期最大值位置"""
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_argmax(X.to_numpy(), p), index=X.index, columns=X.columns
    )


def ts_argmin(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """过去p期最小值位置"""
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_argmin(X.to_numpy(), p), index=X.index, columns=X.columns
    )


def ts_skew(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """过去p期偏度"""
    _check_input(X)
    return pd.DataFrame(_nb._ts_skew(X.to_numpy(), p), index=X.index, columns=X.columns)


def ts_co_skew(X: pd.DataFrame, Y: pd.DataFrame, p: int) -> pd.DataFrame:
    """CoSkewness(X, Y) = E[(Y-EY)(X-EX)^2] / (std(X) * std(Y)^2)"""
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._ts_co_skew(X.to_numpy(), Y.to_numpy(), p),
        index=X.index,
        columns=X.columns,
    )


def ts_kurt(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """过去p期峰度"""
    _check_input(X)
    return pd.DataFrame(_nb._ts_kurt(X.to_numpy(), p), index=X.index, columns=X.columns)


def ts_co_kurt(X: pd.DataFrame, Y: pd.DataFrame, p: int) -> pd.DataFrame:
    """CoKurtosis(X, Y) = E[(Y-EY)(X-EX)^3] / (std(X) * std(Y)^3)"""
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._ts_co_kurt(X.to_numpy(), Y.to_numpy(), p),
        index=X.index,
        columns=X.columns,
    )


def ts_median(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """过去p期中位数"""
    _check_input(X)
    return X.rolling(p).median(engine="numba")


def ts_prod(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    过去p期乘积
    """
    _check_input(X)
    return pd.DataFrame(_nb._ts_prod(X.to_numpy(), p), index=X.index, columns=X.columns)


def ts_cumprod(X: pd.DataFrame) -> pd.DataFrame:
    """
    累计乘积
    """
    _check_input(X)
    return X.cumprod()


def ts_winsorize(X: pd.DataFrame, p: int, n: int = 3) -> pd.DataFrame:
    """
    过去p期去极值
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_winsorize(X.to_numpy(), p, n), index=X.index, columns=X.columns
    )


def ts_scale(X: pd.DataFrame, p: int, constant: float = 0) -> pd.DataFrame:
    """
    过去p期MinMax归一化
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_scale(X.to_numpy(), p, constant), index=X.index, columns=X.columns
    )


def ts_zscore(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """过去p期z-score标准化"""
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_zscore(X.to_numpy(), p), index=X.index, columns=X.columns
    )


def ts_entropy(X: pd.DataFrame, p: int, buckets: int = 10) -> pd.DataFrame:
    """
    过去p期熵
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_entropy(X.to_numpy(), p, buckets), index=X.index, columns=X.columns
    )


def ts_weighted_decay(X: pd.DataFrame, k: int = 0.5) -> pd.DataFrame:
    """今天和昨天加权和"""
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_weighted_decay(X.to_numpy(), k), index=X.index, columns=X.columns
    )


def ts_decay_linear(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """过去p期线性加权"""
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_decay_linear(X.to_numpy(), p), index=X.index, columns=X.columns
    )


def ts_decay_exp_window(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """span为p的指数加权"""
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_decay_exp_window(X.to_numpy(), p),
        index=X.index,
        columns=X.columns,
    )


def ts_hump(X: pd.DataFrame, hump: int = 0.01) -> pd.DataFrame:
    """
    如果abs(x[t] - x[t-1]) > hump，则x[t] = x[t-1] + hump * sign(x[t] - x[t-1])，否则x[t] = x[t-1]
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_hump(X.to_numpy(), hump), index=X.index, columns=X.columns
    )


def ts_hump_decay(X: pd.DataFrame, p: float = 0, relative: bool = True) -> pd.DataFrame:
    """
    relative==False：如果abs(x[t] - x[t-1]) > p，则x[t] = x[t]，否则x[t] = x[t-1]
    relative==True：如果abs(x[t] - x[t-1]) > p * (x[t] + x[t-1])，则x[t] = x[t]，否则x[t] = x[t-1]
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_hump_decay(X.to_numpy(), p, relative), index=X.index, columns=X.columns
    )


def ts_jump_decay(
    X: pd.DataFrame,
    p: int,
    stddev: bool = True,
    sensitivity: float = 0.5,
    force: float = 0.1,
) -> pd.DataFrame:
    """
    ts_jump_decay(x) = abs(x-ts_delay(x, 1)) > sensitivity * ts_std(x, p) ? ts_delay(x,1) + ts_delta(x, 1) * force: x
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_jump_decay(X.to_numpy(), p, stddev, sensitivity, force),
        index=X.index,
        columns=X.columns,
    )


def ts_last_diff_value(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    最近p期内的不等于当前值的最后一个值
    """
    _check_input(X)
    return pd.DataFrame(
        _nb._ts_last_diff_value(X.to_numpy(), p),
        index=X.index,
        columns=X.columns,
    )


def ts_ffill(X: pd.DataFrame) -> pd.DataFrame:
    """向前填充"""
    _check_input(X)
    return X.ffill()


def ts_bfill(X: pd.DataFrame) -> pd.DataFrame:
    """向后填充"""
    _check_input(X)
    return X.bfill()


def ts_fillna(X: pd.DataFrame, value: float) -> pd.DataFrame:
    """填充缺失值"""
    _check_input(X)
    return X.fillna(value)


########## 截面算子 ##########


def cs_normalize(X: pd.DataFrame, stddev: bool = True) -> pd.DataFrame:
    """截面标准化"""
    _check_input(X)
    return pd.DataFrame(
        _nb._cs_normalize(X.to_numpy(), stddev), index=X.index, columns=X.columns
    )


def cs_rank(
    X: pd.DataFrame, method: str | None = "average", pct: bool | None = True
) -> pd.DataFrame:
    """截面排序"""
    _check_input(X)
    return X.rank(axis=1, pct=pct, method=method)


def cs_winsorize(X: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """截面去极值"""
    _check_input(X)
    return pd.DataFrame(
        _nb._cs_winsorize(X.to_numpy(), n), index=X.index, columns=X.columns
    )


def cs_regression_neut(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """截面回归残差"""
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._cs_regression_neut(X.to_numpy(), Y.to_numpy()),
        index=X.index,
        columns=X.columns,
    )


def cs_regression_proj(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """截面回归预测的y"""
    _check_input(X)
    _check_input(Y)
    _check_shape(X, Y)
    return pd.DataFrame(
        _nb._cs_regression_proj(X.to_numpy(), Y.to_numpy()),
        index=X.index,
        columns=X.columns,
    )
