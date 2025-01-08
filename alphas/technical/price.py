from pandas import DataFrame

from AlphaEngine.alphabase import Alpha
from AlphaEngine.operators import numba as op


class AVGPRICE(Alpha):
    def formula(
        self, open: DataFrame, high: DataFrame, low: DataFrame, close: DataFrame
    ) -> DataFrame:
        alpha = op.add(op.add(op.add(open, high), low), close) / 4
        return alpha


class MEDPRICE(Alpha):
    def formula(self, high: DataFrame, low: DataFrame) -> DataFrame:
        alpha = op.add(high, low) / 2
        return alpha


class TYPPRICE(Alpha):
    def formula(self, high: DataFrame, low: DataFrame, close: DataFrame) -> DataFrame:
        alpha = op.add(op.add(high, low), close) / 3
        return alpha


class WCLPRICEW(Alpha):
    def formula(self, high: DataFrame, low: DataFrame, close: DataFrame) -> DataFrame:
        alpha = op.add(op.add(high, low), close * 2) / 4
        return alpha
