from datetime import datetime

import duckdb as ddb
import pandas as pd

ddb.connect()


class DataLoader:
    def __init__(self, data_dir, timecol: str, start_time: str, end_time: str):
        self.data_dir = data_dir
        self.timecol = timecol
        self.start_time: str = pd.to_datetime(start_time).strftime("%Y-%m-%d")
        self.end_time: str = pd.to_datetime(end_time).strftime("%Y-%m-%d")

    def load(self, field):
        df = ddb.sql(
            f"SELECT * FROM read_parquet('{self.data_dir+field}.parquet') WHERE {self.timecol} BETWEEN '{self.start_time}' AND '{self.end_time}'"
        ).df()
        df.set_index(self.timecol, inplace=True)
        return df
