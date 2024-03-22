import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import KFold


def create_xgboost_classifier(X_train: pd.DataFrame,y_train: pd.DataFrame,
                              cv_para: dict,cv_value: int,cv_scoring: str, 
                              random_state: int, njobs: int)-> XGBClassifier:
    
    xgbc_model = XGBClassifier(random_state=random_state)

    grid_search_xgbc = GridSearchCV(xgbc_model, cv_para, cv=cv_value, scoring=cv_scoring, n_jobs = njobs)
    grid_search_xgbc.fit(X_train, y_train.values.ravel())

    best_xgbc_model = grid_search_xgbc.best_estimator_
    return best_xgbc_model

def create_xgboost_classifier_early_stopping(X_train: pd.DataFrame, y_train: pd.DataFrame, 
                              X_validation: pd.DataFrame, y_validation: pd.DataFrame,
                              cv_para: dict, cv_value: int, cv_scoring: str,
                              random_state: int, njobs: int, 
                              early_stop_times: int) -> XGBClassifier:
    
    xgbc_model = XGBClassifier(random_state = random_state)

    if early_stop_times != 0:
        cv_para.update({
            'eval_metric': ['error'],
            'early_stopping_rounds': [early_stop_times]
        })
        cv = KFold(n_splits = cv_value, shuffle = False)
    else:
        cv = cv_value

    grid_search_xgbc = GridSearchCV(xgbc_model, cv_para, cv = cv, scoring = cv_scoring, n_jobs = njobs, verbose = 2)
    grid_search_xgbc.fit(X_train, y_train.values.ravel(), eval_set = [(X_validation, y_validation.values.ravel())])

    best_xgbc_model = grid_search_xgbc.best_estimator_
    best_iteration = best_xgbc_model.best_iteration
    best_score = best_xgbc_model.best_score

    print(f"Best Iteration: {best_iteration}, Best Score: {best_score}")

    return best_xgbc_model


def apply_xgboost_classifier(df: pd.DataFrame, xgboost_model: XGBClassifier) -> pd.DataFrame:
    return xgboost_model.predict_proba(df)

