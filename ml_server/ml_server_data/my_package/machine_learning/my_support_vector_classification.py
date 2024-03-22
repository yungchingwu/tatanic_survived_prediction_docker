import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def create_SVC(X_train: pd.DataFrame,y_train: pd.DataFrame, 
               cv_para: dict, cv_value: int, cv_scoring: str, 
               random_state: int,max_iter: int, njobs: int)-> SVC:
    
    svc_model = SVC(probability = True, random_state = random_state, max_iter = max_iter)

    grid_search_svc = GridSearchCV(svc_model, cv_para, cv = cv_value, scoring = cv_scoring, n_jobs = njobs)
    grid_search_svc.fit(X_train, y_train.values.ravel())

    best_svc_model = grid_search_svc.best_estimator_
    return best_svc_model

def apply_SVC(df: pd.DataFrame, SVC_model: SVC) -> pd.DataFrame:
    return SVC_model.predict_proba(df)

