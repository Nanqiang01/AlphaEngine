from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd


class Alpha(ABC):
    def __init__(
        self,
        dataloader,
    ):
        self.dataloader = dataloader

    def formula(self, *args):
        pass

    def calculate(self):
        code = self.formula.__code__
        fields = code.co_varnames[1 : code.co_argcount]
        data = [self.dataloader.load(arg) for arg in fields]
        # 调用formula函数，计算alpha值
        return self.formula(*data)
