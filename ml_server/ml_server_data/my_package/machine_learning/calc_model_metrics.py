import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def calc_model_metrics(y_data: pd.DataFrame, y_prob: pd.DataFrame)-> tuple:
    y_train_binary = label_binarize(y_data, classes=[0, 1])
    fpr, tpr, thresholds = roc_curve(y_train_binary.ravel(), y_prob[:, 1])
    
    df_result = pd.DataFrame({"Threshold": thresholds,"TP": 0,"FP": 0,"TN": 0,"FN": 0,"Sen": 0.0,"Spe": 0.0,"Pre": 0.0,"Acc": 0.0,"F": 0.0})
    for threshold in thresholds:
        y_pred_labels = (y_prob[:, 1] > threshold).astype(int)        
        tp_count = np.sum((y_pred_labels == 1) & (y_data["Survived"] == 1))
        df_result.loc[df_result["Threshold"] == threshold, "TP"] = np.sum((y_pred_labels == 1) & (y_data["Survived"] == 1))
        df_result.loc[df_result["Threshold"] == threshold, "FP"] = np.sum((y_pred_labels == 1) & (y_data["Survived"] == 0))
        df_result.loc[df_result["Threshold"] == threshold, "TN"] = np.sum((y_pred_labels == 0) & (y_data["Survived"] == 0))
        df_result.loc[df_result["Threshold"] == threshold, "FN"] = np.sum((y_pred_labels == 0) & (y_data["Survived"] == 1))
        df_result["Sen"] = df_result["TP"] / (df_result["TP"] + df_result["FN"])
        df_result["Spe"] = df_result["TN"] / (df_result["TN"] + df_result["FP"])
        df_result["Pre"] = df_result["TP"] / (df_result["TP"] + df_result["FP"])
        df_result["Acc"] = (df_result["TP"] + df_result["TN"]) / (df_result["TP"] + df_result["FP"] + df_result["TN"] + df_result["FN"])
        df_result["F"] = 2 * (df_result["Sen"] * df_result["Pre"]) / (df_result["Sen"] + df_result["Pre"])

    auc_result = auc(fpr, tpr)
    return df_result,auc_result