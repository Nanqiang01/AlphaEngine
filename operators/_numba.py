import numpy as np
from numba import njit, prange


@njit(nogil=True, cache=True)
def _add(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.add(X, Y)


@njit(nogil=True, cache=True)
def _sub(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.subtract(X, Y)


@njit(nogil=True, cache=True)
def _mul(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.multiply(X, Y)


@njit(nogil=True, cache=True)
def _max(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.fmax(X, Y)


@njit(nogil=True, cache=True)
def _min(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.fmin(X, Y)


@njit(nogil=True, cache=True)
def _div(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in range(N):
        x_array = X[:, col]
        y_array = Y[:, col]
        valid_idx = y_array > 1e-10  # 只对非零元素进行除法
        res[valid_idx, col] = np.divide(x_array[valid_idx], y_array[valid_idx])
    return res


@njit(nogil=True, cache=True)
def _ceiling(X: np.ndarray) -> np.ndarray:
    return np.ceil(X)


@njit(nogil=True, cache=True)
def _floor(X: np.ndarray) -> np.ndarray:
    return np.floor(X)


@njit(nogil=True, cache=True)
def _round(X: np.ndarray, n: int) -> np.ndarray:
    return np.round(X, n)


@njit(nogil=True, cache=True)
def _fraction(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * (np.abs(X) - np.floor(np.abs(X)))


@njit(nogil=True, cache=True)
def _log(X: np.ndarray) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in range(N):
        x_array = X[:, col]
        valid_idx = x_array > 1e-10  # 只对正数取对数
        res[valid_idx, col] = np.log(x_array[valid_idx])
    return res


@njit(nogil=True, cache=True)
def _exp(X: np.ndarray) -> np.ndarray:
    return np.exp(X)


@njit(nogil=True, cache=True)
def _sqrt(X: np.ndarray) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in range(N):
        x_array = X[:, col]
        valid_idx = x_array >= 1e-10  # 只对非负数开方
        res[valid_idx, col] = np.sqrt(x_array[valid_idx])
    return res


@njit(nogil=True, cache=True)
def _pow(X: np.ndarray, a: int) -> np.ndarray:
    return np.power(X, a)


@njit(nogil=True, cache=True)
def _abs(X: np.ndarray) -> np.ndarray:
    return np.abs(X)


@njit(nogil=True, cache=True)
def _sign(X: np.ndarray) -> np.ndarray:
    return np.sign(X)


@njit(nogil=True, cache=True)
def _neg(X: np.ndarray) -> np.ndarray:
    return np.negative(X)


@njit(nogil=True, cache=True)
def _inv(X: np.ndarray) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in range(N):
        x_array = X[:, col]
        valid_idx = x_array != 1e-10  # 只对非零元素取倒数
        res[valid_idx, col] = 1 / x_array[valid_idx]
    return res


@njit(nogil=True, cache=True)
def _sigmoid(X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-X))


@njit(nogil=True, cache=True)
def _signed_power(X: np.ndarray, a: int) -> np.ndarray:
    return np.sign(X) * np.abs(X) ** a


@njit(nogil=True, cache=True)
def _s_log_1p(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * _log(np.abs(X))


@njit(nogil=True, cache=True)
def _cos(X: np.ndarray) -> np.ndarray:
    return np.cos(X)


@njit(nogil=True, cache=True)
def _sin(X: np.ndarray) -> np.ndarray:
    return np.sin(X)


@njit(nogil=True, cache=True, parallel=True)
def _tan(X: np.ndarray) -> np.ndarray:
    _, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        x_array = X[:, col]
        valid_idx = np.abs(x_array - np.pi / 2) > 1e-10
        res[valid_idx, col] = np.tan(x_array[valid_idx])
    return res


@njit(nogil=True, cache=True)
def _nan_mask(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.where(Y < 0, np.nan, X)


@njit(nogil=True, cache=True)
def _nan_out(X: np.ndarray, lower: int, upper: int) -> np.ndarray:
    return np.where((X < lower) | (X > upper), np.nan, X)


@njit(nogil=True, cache=True)
def _to_nan(X: np.ndarray, value: int, reverse: bool) -> np.ndarray:
    if reverse:
        return np.where(X == value, np.nan, X)
    else:
        return np.where(X == np.nan, value, X)


@njit(nogil=True, cache=True)
def _purify(X: np.ndarray) -> np.ndarray:
    return np.where(np.isinf(X), np.nan, X)


@njit(nogil=True, cache=True)
def _and(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.logical_and(X, Y)


@njit(nogil=True, cache=True)
def _or(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.logical_or(X, Y)


@njit(nogil=True, cache=True)
def _equal(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.equal(X, Y)


@njit(nogil=True, cache=True)
def _negate(X: np.ndarray) -> np.ndarray:
    return np.logical_not(X)


@njit(nogil=True, cache=True)
def _less(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.less(X, Y)


@njit(nogil=True, cache=True)
def _if_then_else(
    condition: np.ndarray, out1: np.ndarray, out2: np.ndarray
) -> np.ndarray:
    """条件判断"""
    return np.where(condition, out1, out2)


@njit(nogil=True, cache=True)
def _is_not_nan(X: np.ndarray) -> np.ndarray:
    return ~np.isnan(X)


@njit(nogil=True, cache=True)
def _is_nan(X: np.ndarray) -> np.ndarray:
    return np.isnan(X)


@njit(nogil=True, cache=True)
def _is_finite(X: np.ndarray) -> np.ndarray:
    return np.isfinite(X)


@njit(nogil=True, cache=True)
def _is_not_finite(X: np.ndarray) -> np.ndarray:
    return ~np.isfinite(X)


@njit(nogil=True, cache=True, parallel=True)
def _time_from_last_change(X: np.ndarray) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        last_change = np.nan
        for t in range(T):
            if X[t, col] != last_change:
                last_change = X[t, col]
                res[t, col] = 0
            else:
                res[t, col] = res[t - 1, col] + 1
    return res


@njit(nogil=True, cache=True)
def _ts_delay(X: np.ndarray, p: int) -> np.ndarray:
    T, _ = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    res[p:] = X[: T - p]
    return res


@njit(nogil=True, cache=True)
def _ts_delta(X: np.ndarray, p: int) -> np.ndarray:
    T, _ = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    res[p:] = X[p:] - X[: T - p]
    return res


@njit(nogil=True, cache=True)
def _ts_delta_pct(X: np.ndarray, p: int) -> np.ndarray:
    T, _ = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    res[p:] = _div(X[p:], X[: T - p]) - 1
    return res


@njit(nogil=True, cache=True)
def _ts_log_diff(X: np.ndarray, p: int) -> np.ndarray:
    T, _ = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    res[p:] = _log(X[p:]) - _log(X[: T - p])
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_av_diff(X: np.ndarray, p: int) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T):
            res[t, col] = X[t, col] - np.nanmean(X[t - p : t, col])
    return res


@njit(nogil=True, cache=True)
def _ts_cumsum(X: np.ndarray) -> np.ndarray:
    return np.cumsum(X, axis=0)


@njit(nogil=True, cache=True, parallel=True)
def _ts_cov(X: np.ndarray, Y: np.ndarray, p: int) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            y_array = Y[t - p : t, col]
            valid = np.isfinite(x_array) & np.isfinite(y_array)
            n = np.sum(valid)
            if n >= p * 0.8:  # 保留80%以上的有效数据
                x_valid = x_array[valid]
                y_valid = y_array[valid]
                # 手动计算协方差
                x_mean = np.mean(x_valid)
                y_mean = np.mean(y_valid)
                cov = np.sum((x_valid - x_mean) * (y_valid - y_mean)) / (p - 1)
                res[t - 1, col] = cov
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_corr(X: np.ndarray, Y: np.ndarray, p: int) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            y_array = Y[t - p : t, col]
            valid = np.isfinite(x_array) & np.isfinite(y_array)
            n = np.sum(valid)
            if n >= p * 0.8:  # 保留80%以上的有效数据
                x_valid = x_array[valid]
                y_valid = y_array[valid]
                # 计算均值
                x_mean = np.mean(x_valid)
                y_mean = np.mean(y_valid)
                # 计算标准差(防止除零)
                x_std = np.sqrt(np.sum((x_valid - x_mean) ** 2) / (p - 1))
                y_std = np.sqrt(np.sum((y_valid - y_mean) ** 2) / (p - 1))
                # 安全计算相关系数
                if x_std > 1e-10 and y_std > 1e-10:
                    # 计算协方差
                    cov = np.nansum((x_valid - x_mean) * (y_valid - y_mean)) / (p - 1)
                    # 计算相关系数
                    corr = cov / (x_std * y_std)
                    res[t - 1, col] = corr
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_triple_corr(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, p: int) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            y_array = Y[t - p : t, col]
            z_array = Z[t - p : t, col]
            valid = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(z_array)
            n = np.sum(valid)
            if n >= p * 0.8:
                x_valid = x_array[valid]
                y_valid = y_array[valid]
                z_valid = z_array[valid]
                # 计算均值
                x_mean = np.mean(x_valid)
                y_mean = np.mean(y_valid)
                z_mean = np.mean(z_valid)
                # 计算标准差(防止除零)
                x_std = np.sqrt(np.sum((x_valid - x_mean) ** 2) / (p - 1))
                y_std = np.sqrt(np.sum((y_valid - y_mean) ** 2) / (p - 1))
                z_std = np.sqrt(np.sum((z_valid - z_mean) ** 2) / (p - 1))
                # 安全计算相关系数
                if x_std > 1e-10 and y_std > 1e-10 and z_std > 1e-10:
                    # 计算协方差
                    cov = np.sum(
                        (x_valid - x_mean) * (y_valid - y_mean) * (z_valid - z_mean)
                    ) / (p - 1)
                    # 计算相关系数
                    corr = cov / (x_std * y_std * z_std)
                    res[t - 1, col] = corr
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_partial_corr(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, p: int) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            y_array = Y[t - p : t, col]
            z_array = Z[t - p : t, col]
            valid = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(z_array)
            n = np.sum(valid)
            if n >= p * 0.8:
                x_valid = x_array[valid]
                y_valid = y_array[valid]
                z_valid = z_array[valid]
                # 计算均值
                x_mean = np.mean(x_valid)
                y_mean = np.mean(y_valid)
                z_mean = np.mean(z_valid)
                # 计算标准差(防止除零)
                x_std = np.sqrt(np.sum((x_valid - x_mean) ** 2) / (p - 1))
                y_std = np.sqrt(np.sum((y_valid - y_mean) ** 2) / (p - 1))
                z_std = np.sqrt(np.sum((z_valid - z_mean) ** 2) / (p - 1))
                # 安全计算相关系数
                if x_std > 1e-10 and y_std > 1e-10 and z_std > 1e-10:
                    # 计算协方差
                    corr_xy = (
                        np.sum((x_valid - x_mean) * (y_valid - y_mean))
                        / (p - 1)
                        / (x_std * y_std)
                    )
                    corr_xz = (
                        np.sum((x_valid - x_mean) * (z_valid - z_mean))
                        / (p - 1)
                        / (x_std * z_std)
                    )
                    corr_yz = (
                        np.sum((y_valid - y_mean) * (z_valid - z_mean))
                        / (p - 1)
                        / (y_std * z_std)
                    )
                    # 计算偏相关系数
                    partial_corr = (corr_xy - corr_xz * corr_yz) / np.sqrt(
                        (1 - corr_xz**2) * (1 - corr_yz**2)
                    )
                    res[t - 1, col] = partial_corr
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_rank(X: np.ndarray, p: int, method: str, pct: bool) -> np.ndarray:  # noqa: C901
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)

    for col in prange(N):
        for t in range(p, T + 1):
            window = X[t - p : t, col]
            # 过滤有效值
            mask = np.isfinite(window)
            valid_data = window[mask]
            if len(valid_data) > 1e-10:
                # 计算排名
                ranks = np.zeros_like(valid_data)
                if method == "min":
                    # 最小排名方法
                    for i in range(len(valid_data)):
                        ranks[i] = np.sum(valid_data < valid_data[i]) + 1
                elif method == "max":
                    # 最大排名方法
                    for i in range(len(valid_data)):
                        ranks[i] = np.sum(valid_data <= valid_data[i])
                elif method == "average":
                    # 平均排名方法
                    for i in range(len(valid_data)):
                        smaller = np.sum(valid_data < valid_data[i])
                        equal = np.sum(valid_data == valid_data[i])
                        ranks[i] = smaller + (equal + 1) / 2
                elif method == "first":
                    # 首次出现排名
                    unique_vals = np.unique(valid_data)
                    for i in range(len(valid_data)):
                        ranks[i] = np.where(unique_vals == valid_data[i])[0][0] + 1
                elif method == "dense":
                    # 紧凑排名
                    unique_vals = np.unique(valid_data)
                    for i in range(len(valid_data)):
                        ranks[i] = np.where(unique_vals == valid_data[i])[0][0] + 1
                # 归一化到0-1
                if pct:
                    if len(ranks) > 1:
                        ranks = ranks / p
                # 将排名写回结果
                res[t - 1, col] = ranks[-1] if len(ranks) > 0 else np.nan

    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_percentage(X: np.ndarray, p: int, percentage: float) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)

    for col in prange(N):
        for t in range(p, T + 1):
            window = X[t - p : t, col]
            # 过滤有效值
            mask = np.isfinite(window)
            valid_data = window[mask]
            if len(valid_data) > 1e-10:
                res[t - 1, col] = np.percentile(valid_data, percentage * 100)
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_argmax(X: np.ndarray, p: int) -> np.ndarray:
    """过去p期最大值位置"""
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            if np.count_nonzero(np.isnan(x_array)) < p / 5:
                res[t - 1, col] = np.argmax(x_array[::-1])
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_argmin(X: np.ndarray, p: int) -> np.ndarray:
    """过去p期最小值位置"""
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            if np.count_nonzero(np.isnan(x_array)) < p / 5:
                res[t - 1, col] = np.argmin(x_array[::-1])
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_skew(X: np.ndarray, p: int) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T + 1):
            window = X[t - p : t, col]
            # 过滤有效值
            mask = np.isfinite(window)
            valid_data = window[mask]
            n = len(valid_data)
            # 至少需要3个有效值
            if n >= 3:
                # 计算均值
                mean = np.mean(valid_data)
                # 计算标准差(防止除零)
                std = np.std(valid_data)
                if std < 1e-10:
                    res[t - 1, col] = 0.0
                    continue
                # 标准化
                normalized = (valid_data - mean) / std
                # 计算三阶矩(偏度)
                skew = np.mean(normalized**3)
                # 样本偏度修正(Pearson's 偏度系数)
                skew_corrected = skew * ((n * (n - 1)) ** 0.5 / (n - 2))
                res[t - 1, col] = skew_corrected
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_co_skew(X: np.ndarray, Y: np.ndarray, p: int) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T + 1):
            x_window = X[t - p : t, col]
            y_window = Y[t - p : t, col]
            # 过滤有效值
            mask = np.isfinite(x_window) & np.isfinite(y_window)
            valid_data_x = x_window[mask]
            valid_data_y = y_window[mask]
            n = len(valid_data_x)
            # 至少需要3个有效值
            if n >= 3:
                # 计算均值
                x_mean = np.mean(valid_data_x)
                y_mean = np.mean(valid_data_y)
                # 计算标准差(防止除零)
                x_std = np.sqrt(np.sum((valid_data_x - x_mean) ** 2) / (n - 1))
                y_std = np.sqrt(np.sum((valid_data_y - y_mean) ** 2) / (n - 1))
                if x_std < 1e-10 or y_std < 1e-10:
                    res[t - 1, col] = 0.0
                    continue
                # 计算三阶矩(偏度)
                co_skew = (
                    np.sum((valid_data_y - y_mean) * (valid_data_x - x_mean) ** 2)
                    / (n - 1)
                    / (x_std * y_std**2)
                )
                res[t - 1, col] = co_skew
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_kurt(X: np.ndarray, p: int) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T + 1):
            window = X[t - p : t, col]
            # 过滤有效值
            mask = np.isfinite(window)
            valid_data = window[mask]
            n = len(valid_data)
            # 至少需要4个有效值
            if n >= 4:
                # 计算均值
                mean = np.mean(valid_data)
                # 计算标准差(防止除零)
                std = np.std(valid_data)
                if std < 1e-10:
                    res[t - 1, col] = 0.0
                    continue
                # 标准化
                normalized = (valid_data - mean) / std
                # 计算四阶矩(峰度)
                kurt = np.mean(normalized**4)
                # 样本峰度修正
                kurt_corrected = kurt * (
                    ((n + 1) * (n - 1)) / ((n - 2) * (n - 3))
                ) - 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
                res[t - 1, col] = kurt_corrected
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_co_kurt(X: np.ndarray, Y: np.ndarray, p: int) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T + 1):
            x_window = X[t - p : t, col]
            y_window = Y[t - p : t, col]
            # 过滤有效值
            mask = np.isfinite(x_window) & np.isfinite(y_window)
            valid_data_x = x_window[mask]
            valid_data_y = y_window[mask]
            n = len(valid_data_x)
            # 至少需要4个有效值
            if n >= 4:
                # 计算均值
                x_mean = np.mean(valid_data_x)
                y_mean = np.mean(valid_data_y)
                # 计算标准差(防止除零)
                x_std = np.sqrt(np.sum((valid_data_x - x_mean) ** 2) / (n - 1))
                y_std = np.sqrt(np.sum((valid_data_y - y_mean) ** 2) / (n - 1))
                if x_std < 1e-10 or y_std < 1e-10:
                    res[t - 1, col] = 0.0
                    continue
                # 计算四阶矩(峰度)
                co_kurt = (
                    np.sum((valid_data_y - y_mean) * (valid_data_x - x_mean) ** 3)
                    / (n - 1)
                    / (x_std * y_std**3)
                )
                res[t - 1, col] = co_kurt
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_prod(X: np.ndarray, p: int) -> np.ndarray:
    """
    过去p期乘积
    """
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            if np.count_nonzero(np.isnan(x_array)) < p / 5:
                res[t - 1, col] = np.nanprod(x_array)
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_winsorize(X: np.ndarray, p: int, n: int) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            if np.count_nonzero(np.isnan(x_array)) < p / 5:
                mad = np.nanmedian(np.abs(x_array - np.nanmedian(x_array)))
                median = np.nanmedian(x_array)
                res[t - 1, col] = np.clip(
                    x_array, median - 3 * 1.4826 * mad, median + 3 * 1.4826 * mad
                )[-1]
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_scale(X: np.ndarray, p: int, constant: float) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            if np.count_nonzero(~np.isnan(x_array)) > max(2, p / 5):
                x_min = np.nanmin(x_array)
                x_max = np.nanmax(x_array)
                if x_max - x_min < 1e-10:
                    res[t - 1, col] = 1 + constant
                else:
                    res[t - 1, col] = (x_array[-1] - x_min) / (x_max - x_min) + constant
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_zscore(X: np.ndarray, p: int) -> np.ndarray:
    """过去p期z-score标准化"""
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            if np.count_nonzero(~np.isnan(x_array)) > max(2, p / 5):
                x_mean = np.nanmean(x_array)
                x_std = np.sqrt(np.nansum((x_array - x_mean) ** 2) / (p - 1))
                if x_std < 1e-10:
                    res[t - 1, col] = 0.0
                else:
                    res[t - 1, col] = (x_array[-1] - x_mean) / x_std
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_entropy(X: np.ndarray, p: int, buckets: int) -> np.ndarray:
    """
    过去p期熵
    """
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T + 1):
            x_array = X[t - p : t, col]
            if np.count_nonzero(~np.isnan(x_array)) > max(2, p / 5):
                # 计算直方图
                hist, _ = np.histogram(
                    x_array,
                    bins=buckets,
                    range=(np.nanmin(x_array), np.nanmax(x_array)),
                )
                # 计算熵
                res[t - 1, col] = np.log(p) - 1 / np.nansum(hist * _log(hist))
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_weighted_decay(X: np.ndarray, k: int) -> np.ndarray:
    """今天和昨天加权和"""
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    weights = np.array([1 - k, k])
    weights = weights / np.sum(weights)

    for col in prange(N):
        for t in range(1, T):
            arr = X[t - 1 : t + 1, col]
            res[t, col] = np.nansum(arr * weights)

    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_decay_linear(X: np.ndarray, p: int) -> np.ndarray:
    """过去p期线性加权"""
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    weights = np.arange(p, 0, -1)
    weights = weights / np.sum(weights)

    for col in prange(N):
        for t in range(p, T + 1):
            arr = X[t - p : t, col]
            res[t - 1, col] = np.nansum(arr * weights)

    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_decay_exp_window(X: np.ndarray, p: int) -> np.ndarray:
    """span为p的指数加权"""
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    alpha = 2 / (p + 1)
    for col in prange(N):
        valid_count = 0
        last_avg = np.nan
        denominator = 1
        for t in range(T):
            value = X[t, col]
            if not np.isnan(value):
                valid_count += 1
                if valid_count == 1:
                    last_avg = value
                    denominator = 1
                else:
                    denominator = denominator * (1 - alpha) + 1
                    last_avg = (1 - alpha) * last_avg + value
                res[t, col] = last_avg / denominator
            else:
                res[t, col] = np.nan
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_hump(X: np.ndarray, hump: float = 0.01) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(1, T):
            diff = X[t, col] - X[t - 1, col]
            res[t, col] = (
                X[t, col]
                if np.abs(diff) <= hump
                else X[t - 1, col] + hump * np.sign(diff)
            )
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_hump_decay(X: np.ndarray, p: float = 0, relative: bool = True) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(1, T):
            diff = X[t, col] - X[t - 1, col]
            if relative:
                hump = p * (X[t, col] + X[t - 1, col])
            else:
                hump = p
            res[t, col] = X[t, col] if np.abs(diff) <= hump else X[t - 1, col]
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_jump_decay(
    X: np.ndarray, p: int, stddev: bool, sensitivity: float, force: float
) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(1, T):
            diff = X[t, col] - X[t - 1, col]
            if stddev:
                std = np.nanstd(X[t - p : t, col])
                jump = sensitivity * std
            else:
                jump = sensitivity
            res[t, col] = (
                X[t - 1, col] + diff * force if np.abs(diff) > jump else X[t, col]
            )
    return res


@njit(nogil=True, cache=True, parallel=True)
def _ts_last_diff_value(X: np.ndarray, p: int) -> np.ndarray:
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for col in prange(N):
        for t in range(p, T):
            for i in range(1, p + 1):
                if X[t, col] != X[t - i, col]:
                    res[t, col] = X[t - i, col]
                    break
    return res


@njit(nogil=True, cache=True, parallel=True)
def _cs_normalize(X: np.ndarray, stddev: bool) -> np.ndarray:
    """截面标准化"""
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for t in range(T):
        x_array = X[t, :]
        if stddev:
            mean = np.nanmean(x_array)
            std = np.sqrt(
                np.nansum((x_array - mean) ** 2) / (np.sum(np.isfinite(x_array)) - 1)
            )
            if std < 1e-10:
                res[t, :] = 0.0
            else:
                res[t, :] = (x_array - mean) / std
        else:
            res[t, :] = x_array - np.nanmin(x_array)
    return res


@njit(nogil=True, cache=True)
def _cs_winsorize(X: np.ndarray, n: int) -> np.ndarray:
    """截面去极值"""
    T, N = X.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for t in range(T):
        valid_idx = ~np.isnan(X[t])
        x_array = X[t, valid_idx]
        # 使用3MAD法去极值
        median = np.nanmedian(x_array)
        mad = np.nanmedian(np.abs(x_array - median))
        upper = median + n * 1.4826 * mad
        lower = median - n * 1.4826 * mad
        res[t, valid_idx] = np.clip(x_array, lower, upper)
    return res


@njit(nogil=True, cache=True)
def _cs_regression_neut(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """截面回归残差"""
    T, N = Y.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for t in range(T):
        y_row = Y[t, :]
        x_row = X[t, :]
        valid_y = ~np.isnan(y_row)
        valid_x = ~np.isnan(x_row)
        if np.count_nonzero(valid_y) > 2 and np.count_nonzero(valid_x) > 2:
            y = y_row[valid_y]
            x = x_row[valid_x]
            y_demean = y - np.nanmean(y)
            x_demean = x - np.nanmean(x)
            beta = np.nansum(y_demean * x_demean) / np.nansum(x_demean**2)
            res[t, valid_y] = y_demean - beta * x_demean

    return res


@njit(nogil=True, cache=True)
def _cs_regression_proj(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """截面回归预测的y"""
    T, N = Y.shape
    res = np.full_like(X, np.nan, dtype=np.float64)
    for t in range(T):
        y_row = Y[t, :]
        x_row = X[t, :]
        valid_y = ~np.isnan(y_row)
        valid_x = ~np.isnan(x_row)
        if np.count_nonzero(valid_y) > 2 and np.count_nonzero(valid_x) > 2:
            y = y_row[valid_y]
            x = x_row[valid_x]
            y_demean = y - np.nanmean(y)
            x_demean = x - np.nanmean(x)
            beta = np.nansum(y_demean * x_demean) / np.nansum(x_demean**2)
            res[t, valid_y] = np.nanmean(y) + beta * x_demean

    return res
