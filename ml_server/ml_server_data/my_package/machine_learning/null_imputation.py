import pandas as pd


class null_imputation:
    def __init__(self, null_imputation_labels: list = None, null_imputation_dict: dict = None):
        self.null_imputation_labels = null_imputation_labels if null_imputation_labels is not None else []
        self.null_imputation_dict = null_imputation_dict if null_imputation_dict is not None else {}

    def create_null_imputation(self, df: pd.DataFrame, imputation_labels: list , imputation_type: str) -> dict:
        self.null_imputation_labels = imputation_labels.copy()
        if imputation_type == "mean":
            return self.create_null_imputation_mean(df)
        elif imputation_type == "median":
            return self.create_null_imputation_median(df)
        else:
            return self.create_null_imputation_mean(df)

    def create_null_imputation_mean(self, df: pd.DataFrame) -> dict:
        null_imputation_dict = {}
        for label in self.null_imputation_labels:
            null_imputation_dict[label] = df[label].mean()
        
        self.null_imputation_dict = null_imputation_dict
        return null_imputation_dict

    def create_null_imputation_median(self, df: pd.DataFrame) -> dict:
        null_imputation_dict = {}
        for label in self.null_imputation_labels:
            null_imputation_dict[label] = df[label].median()
        
        self.null_imputation_dict = null_imputation_dict
        return null_imputation_dict

    def apply_null_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for column, value in self.null_imputation_dict.items():
            if column not in df_copy.columns:
                df_copy[column] = value
            else:
                if isinstance(value, (int, float)):
                    df_copy[column].fillna(value, inplace=True)
                else:
                    df_copy[column] = df_copy[column].apply(lambda x: int(x) if str(x).isdigit() else value)
                    df_copy[column].fillna(value, inplace=True)
        return df_copy




