import pandas as pd

class data_filter:
    def __init__(self):
        self.filter = None

    def create_data_filter(self, filter: list)-> list:
        self.filter = filter
        return self.filter

    def apply_data_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.filter == None:
            return df
        else:
            return df[self.filter]
