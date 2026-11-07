"""
Chọn F2-score với Beta = 2 để ưu tiên recall.
First idea for metrics, not yet been proven optimal
"""

import os

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def load_split_data(
    dataset_type: str = "full",
    data_dir: str = "../data/processed",
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    helper func để load data chung cho cả nhóm
    dataset_type: 'full' hoặc 'selected'
    """
    filename = (
        "diabetes_full.csv"
        if dataset_type.lower() == "full"
        else "diabetes_selected.csv"
    )
    filepath = os.path.join(data_dir, filename)

    df = pd.read_csv(filepath)
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def evaluate_model(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    threshold: float = 0.5,
    model_name: str = "Model",
    dataset_name: str = "Full",
):
    """
    Tính toán bộ chỉ số đánh giá cho bài toán
    - y_true: Nhãn thực tế (0 hoặc 1)
    - y_prob: Xác suất dự đoán lớp 1 (từ model.predict_proba(X_test)[:, 1])
    - threshold: Ngưỡng cắt xác suất (mặc định 0.5). Recommend chạy find_best_threshold trước để tìm threshold đạt F2 tốt nhất
    rồi mới evaluate model
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # độ đặc hiệu bệnh lý (True Negative rate) - người khỏe mạnh thực sự
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics_dict = {
        "Model": model_name,
        "Dataset": dataset_name,
        "Threshold": round(float(threshold), 3),
        "PR_AUC": round(float(average_precision_score(y_true, y_prob)), 4),
        "ROC_AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "F2_Score": round(
            float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 4
        ),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "F1_Score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "Specificity (TNR)": round(specificity, 4),
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
    }
    return metrics_dict


def find_best_threshold(
    y_true: ArrayLike, y_prob: ArrayLike, beta: float = 2.0
) -> tuple[float, float]:
    """
    threshold finder based on F2-score (khoảng tử 0.05 - 0.9)
    """
    y_prob_arr = np.asarray(y_prob)
    thresholds = np.arange(0.05, 0.9, 0.01)
    best_thre = 0.50
    best_score = -1.0
    for thre in thresholds:
        y_pred = (y_prob_arr >= thre).astype(int)
        score = float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))
        if score > best_score:
            best_score = score
            best_thre = float(thre)
    return round(best_thre, 2), round(best_score, 4)
