import numpy as np
import pandas as pd
import torch


def _check_input(X):
    """检查输入类型"""
    if isinstance(X, pd.DataFrame):
        return X
    else:
        raise ValueError("输入类型错误!")


def _check_shape(X, Y):
    """检查矩阵大小是否一致"""
    if isinstance(X, pd.DataFrame) and isinstance(Y, pd.DataFrame):
        if np.any(X.index != Y.index) or np.any(X.columns != Y.columns):
            raise ValueError("X和Y的index或columns不一致!")
    else:
        raise ValueError("X和Y的类型不一致!")


def add(X, Y):
    """加法"""
    X, Y = _check_input(X), _check_input(Y)
    _check_shape(X, Y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device)
    Y_tensor = torch.tensor(Y.to_numpy(), device=device)
    result = X_tensor + Y_tensor
    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


def sub(X, Y):
    """减法"""
    X, Y = _check_input(X), _check_input(Y)
    _check_shape(X, Y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device)
    Y_tensor = torch.tensor(Y.to_numpy(), device=device)
    result = X_tensor - Y_tensor
    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


def mul(X, Y):
    """乘法"""
    X, Y = _check_input(X), _check_input(Y)
    _check_shape(X, Y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device)
    Y_tensor = torch.tensor(Y.to_numpy(), device=device)
    result = X_tensor * Y_tensor
    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


def div(X, Y):
    """安全除法"""
    X, Y = _check_input(X), _check_input(Y)
    _check_shape(X, Y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device, dtype=torch.float32)
    Y_tensor = torch.tensor(Y.to_numpy(), device=device, dtype=torch.float32)
    result = torch.where(
        Y_tensor != 0, X_tensor / Y_tensor, torch.tensor(float("nan"), device=device)
    )
    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


def log(X):
    """对数"""
    X = _check_input(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device)
    result = torch.log(X_tensor)
    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


def exp(X):
    """指数"""
    X = _check_input(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device)
    result = torch.exp(X_tensor)
    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


def sqrt(X):
    """开方"""
    X = _check_input(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device)
    result = torch.sqrt(X_tensor)
    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


def pow(X, a):
    """幂"""
    X = _check_input(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device)
    result = torch.pow(X_tensor, a)
    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


def abs(X):
    """绝对值"""
    X = _check_input(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device)
    result = torch.abs(X_tensor)
    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


def sign(X):
    """符号函数"""
    X = _check_input(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device)
    result = torch.sign(X_tensor)
    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


def neg(X):
    """取负"""
    X = _check_input(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device)
    result = -X_tensor
    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


def inv(X):
    """取倒数"""
    X = _check_input(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device, dtype=torch.float32)
    result = torch.where(
        X_tensor != 0, 1 / X_tensor, torch.tensor(float("nan"), device=device)
    )
    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


# 截面算子
def cs_rank(X):
    """截面排序"""
    X = _check_input(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device)

    # 每行排序并归一化
    result = torch.argsort(torch.argsort(X_tensor, dim=1), dim=1).float() / (
        X_tensor.shape[1] - 1
    )

    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


def cs_winsorize(X, n=3):
    """截面去极值"""
    X = _check_input(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device, dtype=torch.float32)

    result = torch.empty_like(X_tensor)
    for t in range(X_tensor.shape[0]):
        row = X_tensor[t]
        valid_mask = ~torch.isnan(row)
        valid_values = row[valid_mask]

        if valid_values.numel() > 0:
            median = torch.nanmedian(valid_values)
            mad = torch.nanmedian(torch.abs(valid_values - median))
            upper = median + n * 1.4826 * mad
            lower = median - n * 1.4826 * mad

            clipped_values = torch.clamp(valid_values, lower, upper)
            result[t, valid_mask] = clipped_values
            result[t, ~valid_mask] = float("nan")
        else:
            result[t] = float("nan")

    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


# 时间序列算子
def ts_decay_linear(X, p):
    """过去p期线性加权"""
    X = _check_input(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device)

    T, N = X_tensor.shape
    result = torch.empty_like(X_tensor)

    weights = torch.arange(p, 0, -1, device=device).float()
    weights /= weights.sum()

    for col in range(N):
        for t in range(p, T + 1):
            arr = X_tensor[t - p : t, col]
            result[t - 1, col] = torch.nansum(arr * weights)

    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


def ts_decay_exp_window(X, p):
    """span为p的指数加权"""
    X = _check_input(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.to_numpy(), device=device)

    T, N = X_tensor.shape
    result = torch.empty_like(X_tensor)

    alpha = 2 / (p + 1)

    for col in range(N):
        valid_count = 0
        last_avg = torch.tensor(float("nan"), device=device)

        for t in range(T):
            value = X_tensor[t, col]
            if not torch.isnan(value):
                valid_count += 1
                if valid_count == 1:
                    last_avg = value
                else:
                    last_avg = (1 - alpha) * last_avg + value
                result[t, col] = last_avg
            else:
                result[t, col] = float("nan")

    return pd.DataFrame(result.cpu().numpy(), index=X.index, columns=X.columns)


# 常数算子
def get1():
    return torch.tensor(1)


def get5():
    return torch.tensor(5)


def get10():
    return torch.tensor(10)


def get20():
    return torch.tensor(20)


def get60():
    return torch.tensor(60)


def get122():
    return torch.tensor(122)


def get244():
    return torch.tensor(244)
