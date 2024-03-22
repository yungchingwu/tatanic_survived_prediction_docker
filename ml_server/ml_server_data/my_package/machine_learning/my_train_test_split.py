import pandas as pd
from sklearn.model_selection import train_test_split


def my_train_test_split(df: pd.DataFrame, X_label: list, y_label: list, test_size: float, random_state: int) -> tuple:
    df_x = df[X_label]
    df_y = df[y_label]  
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size = test_size, random_state = random_state)
    return X_train, X_test, y_train, y_test