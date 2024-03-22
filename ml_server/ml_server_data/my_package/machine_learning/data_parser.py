import pandas as pd


class data_parser:
    def __init__(self, data_parser_dict=None):
        self.data_parser_dict = data_parser_dict

    def create_data_parser_dict(self, data_parser_dict):
        self.data_parser_dict = data_parser_dict
    
    def apply_data_parser(self, df: pd.DataFrame):
        if self.data_parser_dict is None:
            print("Data parser dict is not provided.")
            return None
        
        for column, dtype in self.data_parser_dict.items():
            if column not in df.columns:
                print(f"Column '{column}' not found in the DataFrame. Skipping...")
                continue
            
            try:
                df.loc[:, column] = df[column].astype(dtype)
            except ValueError:
                print(f"Error converting values in column '{column}' to {dtype}. Setting invalid values to NaN.")
                df[column] = pd.to_numeric(df[column], errors='coerce')

        return df

