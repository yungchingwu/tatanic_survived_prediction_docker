import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def create_random_forest_classifier(X_train: pd.DataFrame, y_train: pd.DataFrame, 
                                    cv_para: dict, cv_value: int, cv_scoring: str, 
                                    random_state: int, njobs: int)-> RandomForestClassifier:
    
    rf_model = RandomForestClassifier(random_state = random_state)

    grid_search_rf = GridSearchCV(rf_model, cv_para, cv = cv_value, scoring = cv_scoring, n_jobs = njobs)
    grid_search_rf.fit(X_train, y_train.values.ravel())

    best_rf_model = grid_search_rf.best_estimator_
    return best_rf_model

def apply_random_forest_classifier(df: pd.DataFrame, random_forest_classifier_model: RandomForestClassifier) -> pd.DataFrame:
    return random_forest_classifier_model.predict_proba(df)
