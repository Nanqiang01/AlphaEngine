from pandas import DataFrame

from AlphaEngine.alphabase import Alpha
from AlphaEngine.operators import numba as op


class mmt_normal_M(Alpha):
    """过去p期的对数收益率"""

    name = "收益率"

    def formula(self, close: DataFrame) -> DataFrame:
        timeperiod = 20
        return op.ts_log_diff(close, timeperiod)


class mmt_avg(Alpha):
    """当期收盘价 / 过去p期均价 - 1"""

    name = "相对均价的收益率"

    def formula(self, close: DataFrame) -> DataFrame:
        timeperiod = 20
        return op.div(close, op.ts_mean(close, timeperiod)) - 1


class mmt_intraday(Alpha):
    """日内涨跌幅之和"""

    name = "日内动量"

    def formula(self, open: DataFrame, close: DataFrame) -> DataFrame:
        timeperiod = 20
        return op.ts_sum(op.div(open, close), timeperiod)


class mmt_overnight(Alpha):
    """隔夜涨跌幅之和"""

    name = "隔夜动量"

    def formula(self, open: DataFrame, close: DataFrame) -> DataFrame:
        timeperiod = 20
        return op.ts_sum(op.div(op.ts_delay(close, 1), open), timeperiod)


class mmt_range(Alpha):
    """振幅大的前20%的收盘收益率 - 振幅小的后20%的收盘收益率"""

    name = "振幅调整动量"

    def formula(self, high: DataFrame, low: DataFrame, close: DataFrame) -> DataFrame:
        timeperiod = 252
        amplitude = op.sub(high, low)
        returns = op.ts_log_diff(close, timeperiod)
        amp_20 = op.ts_percentage(amplitude, timeperiod, 0.2)
        amp_80 = op.ts_percentage(amplitude, timeperiod, 0.8)
        returns_20 = op.if_then_else(amplitude > amp_20, returns, 0)
        returns_80 = op.if_then_else(amplitude < amp_80, returns, 0)
        return op.sub(
            op.ts_sum(returns_20, timeperiod), op.ts_sum(returns_80, timeperiod)
        )


class mmt_route(Alpha):
    """过去p期内收益率 / 过去p期内日度涨跌幅绝对值之和"""

    name = "路径调整动量"

    def formula(self, close: DataFrame) -> DataFrame:
        returns = op.ts_log_diff(close, 20)
        abs_returns = op.ts_sum(op.abs(op.ts_log_diff(close, 1)), 20)
        return op.div(returns, abs_returns)
