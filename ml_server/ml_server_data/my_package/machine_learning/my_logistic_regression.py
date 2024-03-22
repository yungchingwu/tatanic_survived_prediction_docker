import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def create_logistic_regression(X_train: pd.DataFrame, y_train: pd.DataFrame, 
                               cv_para: dict, cv_value: int, cv_scoring: str, 
                               random_state: int, max_iter: int, njobs: int)-> LogisticRegression:
    
    lr_model = LogisticRegression(random_state = random_state, max_iter = max_iter)

    grid_search_lr = GridSearchCV(lr_model, cv_para, cv = cv_value, scoring = cv_scoring, n_jobs = njobs)
    grid_search_lr.fit(X_train, y_train.values.ravel())

    best_lr_model = grid_search_lr.best_estimator_
    return best_lr_model

def apply_logistic_regression(df: pd.DataFrame, logistic_regression_model: LogisticRegression) -> pd.DataFrame:
    return logistic_regression_model.predict_proba(df)
