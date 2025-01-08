from pandas import DataFrame

from AlphaEngine.alphabase import Alpha
from AlphaEngine.operators import numba as op


class AD(Alpha):
    def formula(
        self, high: DataFrame, low: DataFrame, close: DataFrame, volume: DataFrame
    ) -> DataFrame:
        ad = op.div(
            op.sub(op.sub(close, low), op.sub(high, close)),
            op.mul(op.sub(high, low), volume),
        )
        alpha = op.ts_cumsum(ad)
        return alpha


class ADOSC(Alpha):
    def formula(
        self, high: DataFrame, low: DataFrame, close: DataFrame, volume: DataFrame
    ) -> DataFrame:
        fastperiod = 3
        slowperiod = 10
        ad = op.div(
            op.sub(op.sub(close, low), op.sub(high, close)),
            op.mul(op.sub(high, low), volume),
        )
        alpha = op.sub(
            op.ts_decay_exp_window(ad, fastperiod),
            op.ts_decay_exp_window(ad, slowperiod),
        )
        return alpha


class OBV(Alpha):
    def formula(self, close: DataFrame, volume: DataFrame) -> DataFrame:
        obv = op.mul(op.ts_fillna(op.sign(op.ts_delta(close, 1)), 1), volume)
        alpha = op.ts_cumsum(obv)
        return alpha
