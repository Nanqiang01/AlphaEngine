from pandas import DataFrame

from AlphaEngine.alphabase import Alpha
from AlphaEngine.operators import numba as op


class ATR(Alpha):
    def formula(self, high: DataFrame, low: DataFrame, close: DataFrame) -> DataFrame:
        timeperiod = 14
        pre_close = op.ts_delay(close, 1)
        tr1 = op.sub(high, low)
        tr2 = op.abs(op.sub(high, pre_close))
        tr3 = op.abs(op.sub(low, pre_close))
        tr = op.max(op.max(tr1, tr2), tr3)
        alpha = op.ts_mean(tr, timeperiod)
        return alpha


class NATR(Alpha):
    def formula(self, high: DataFrame, low: DataFrame, close: DataFrame) -> DataFrame:
        timeperiod = 14
        pre_close = op.ts_delay(close, 1)
        tr1 = op.sub(high, low)
        tr2 = op.abs(op.sub(high, pre_close))
        tr3 = op.abs(op.sub(low, pre_close))
        tr = op.max(op.max(tr1, tr2), tr3)
        alpha = op.ts_mean(tr, timeperiod) / close
        return alpha
