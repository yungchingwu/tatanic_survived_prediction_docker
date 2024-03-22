import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def apply_data_balance(df_X: pd.DataFrame, df_y: pd.DataFrame, random_state: int, balance_type: str) -> tuple:
    if balance_type == 'under_sampling':
        return data_balance_under_sampling(df_X, df_y, random_state)
    elif balance_type == 'over_sampling':
        return data_balance_over_sampling(df_X, df_y, random_state)
    else:
        return data_balance_under_sampling(df_X, df_y, random_state)
    
def data_balance_under_sampling(df_X: pd.DataFrame, df_y: pd.DataFrame, random_state: int) -> tuple:
    under_sampler = RandomUnderSampler(random_state = random_state)
    X_under_sampling, y_under_sampleing= under_sampler.fit_resample(df_X, df_y)
    return X_under_sampling, y_under_sampleing

def data_balance_over_sampling(df_X: pd.DataFrame, df_y: pd.DataFrame, random_state: int) -> tuple:
    smote = SMOTE(random_state = random_state)
    X_smote, y_smote = smote.fit_resample(df_X, df_y)
    return X_smote, y_smote
