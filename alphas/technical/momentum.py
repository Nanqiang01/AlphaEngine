from pandas import DataFrame

from AlphaEngine.alphabase import Alpha
from AlphaEngine.operators import numba as op


class APO_SMA(Alpha):
    def formula(self, close: DataFrame) -> DataFrame:
        fastperiod = 12
        slowperiod = 26
        alpha = op.sub(op.ts_mean(close, fastperiod), op.ts_mean(close, slowperiod))
        return alpha


class APO_EMA(Alpha):
    def formula(self, close: DataFrame) -> DataFrame:
        fastperiod = 12
        slowperiod = 26
        alpha = op.sub(
            op.ts_decay_exp_window(close, fastperiod),
            op.ts_decay_exp_window(close, slowperiod),
        )
        return alpha


class AROON_aroondown(Alpha):
    def formula(self, low: DataFrame) -> DataFrame:
        timeperiod = 14
        alpha = 1 - op.ts_arg_min(low, timeperiod) / timeperiod
        return alpha


class AROON_aroonup(Alpha):
    def formula(self, high: DataFrame) -> DataFrame:
        timeperiod = 14
        alpha = 1 - op.ts_arg_max(high, timeperiod) / timeperiod
        return alpha


class MACD_macd(Alpha):
    def formula(self, close: DataFrame) -> DataFrame:
        fastperiod = 12
        slowperiod = 26
        alpha = op.sub(
            op.ts_decay_exp_window(close, fastperiod),
            op.ts_decay_exp_window(close, slowperiod),
        )
        return alpha


class MACD_macdhist(Alpha):
    def formula(self, close: DataFrame) -> DataFrame:
        fastperiod = 12
        slowperiod = 26
        signalperiod = 9
        macd = op.sub(
            op.ts_decay_exp_window(close, fastperiod),
            op.ts_decay_exp_window(close, slowperiod),
        )
        signal = op.ts_decay_exp_window(macd, signalperiod)
        alpha = macd - signal
        return alpha


class MACD_macdsignal(Alpha):
    def formula(self, close: DataFrame) -> DataFrame:
        fastperiod = 12
        slowperiod = 26
        signalperiod = 9
        macd = op.sub(
            op.ts_decay_exp_window(close, fastperiod),
            op.ts_decay_exp_window(close, slowperiod),
        )
        alpha = op.ts_decay_exp_window(macd, signalperiod)
        return alpha


class MOM(Alpha):
    def formula(self, close: DataFrame) -> DataFrame:
        timeperiod = 10
        alpha = op.ts_delta(close, timeperiod)
        return alpha


class PPO_SMA(Alpha):
    def formula(self, close: DataFrame) -> DataFrame:
        fastperiod = 12
        slowperiod = 26
        alpha = op.div(op.ts_mean(close, fastperiod), op.ts_mean(close, slowperiod)) - 1
        return alpha


class PPO_EMA(Alpha):
    def formula(self, close: DataFrame) -> DataFrame:
        fastperiod = 12
        slowperiod = 26
        alpha = (
            op.div(
                op.ts_decay_exp_window(close, fastperiod),
                op.ts_decay_exp_window(close, slowperiod),
            )
            - 1
        )
        return alpha


class ROC(Alpha):
    def formula(self, close: DataFrame) -> DataFrame:
        timeperiod = 10
        alpha = op.ts_delta_pct(close, timeperiod)
        return alpha


class RSI(Alpha):
    def formula(self, close: DataFrame) -> DataFrame:
        timeperiod = 14
        span = 2 * timeperiod - 1
        dif = op.ts_fillna(op.ts_delta(close, 1), 0)
        alpha = op.div(
            op.ts_decay_exp_window(op.max(dif, 0), span),
            op.ts_decay_exp_window(op.abs(dif), span),
        )
        return alpha


class STOCHF_fastk(Alpha):
    def formula(self, high: DataFrame, low: DataFrame, close: DataFrame) -> DataFrame:
        fastk_period = 5
        a = op.ts_max(high, fastk_period)
        b = op.ts_min(low, fastk_period)
        alpha = op.div(op.sub(close, b), op.sub(a, b))
        return alpha


class STOCHF_fastd(Alpha):
    def formula(self, high: DataFrame, low: DataFrame, close: DataFrame) -> DataFrame:
        fastk_period = 5
        fastd_period = 3
        a = op.ts_max(high, fastk_period)
        b = op.ts_min(low, fastk_period)
        alpha = op.ts_mean(op.div(op.sub(close, b), op.sub(a, b)), fastd_period)
        return alpha


class TRIX(Alpha):
    def formula(self, close: DataFrame) -> DataFrame:
        timeperiod = 30
        alpha = op.ts_delta_pct(
            op.ts_decay_exp_window(
                op.ts_decay_exp_window(
                    op.ts_decay_exp_window(close, timeperiod), timeperiod
                ),
                timeperiod,
            ),
            1,
        )
        return alpha


class WILLR(Alpha):
    def formula(self, high: DataFrame, low: DataFrame, close: DataFrame) -> DataFrame:
        timeperiod = 14
        a = op.ts_max(high, timeperiod)
        b = op.ts_min(low, timeperiod)
        alpha = op.div(op.sub(a, close), op.sub(a, b))
        return alpha
