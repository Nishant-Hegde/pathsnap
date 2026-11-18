"""
PathSnap Common Metrics Module (Owner: Rishab Naveen)
Standardized evaluation metrics supporting both schema conventions:
- Saanvi Samayam's generated candidate tables: grouping by ['trip_id', 'point'], label in 'y'
- Team contract: grouping by ['trip_id', 'obs_idx'], label in 'label'
"""
from typing import List, Union
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss as sk_log_loss, precision_score, recall_score


def _resolve_cols(df: pd.DataFrame, label_col: str = None, group_cols: List[str] = None):
    # Resolve label column
    if label_col is None:
        if "y" in df.columns:
            label_col = "y"
        elif "label" in df.columns:
            label_col = "label"
        else:
            raise KeyError(f"Neither 'y' nor 'label' found in DataFrame columns: {df.columns.tolist()}")

    # Resolve grouping columns
    if group_cols is None:
        point_col = "point" if "point" in df.columns else ("obs_idx" if "obs_idx" in df.columns else None)
        if "trip_id" in df.columns and point_col:
            group_cols = ["trip_id", point_col]
        elif point_col:
            group_cols = [point_col]
        else:
            raise KeyError(f"Neither 'point' nor 'obs_idx' found in DataFrame columns: {df.columns.tolist()}")

    return label_col, group_cols


def top1_accuracy(
    df: pd.DataFrame,
    prob_col: str = "prob",
    label_col: str = None,
    group_cols: List[str] = None
) -> float:
    """
    Computes Top-1 Accuracy:
    Fraction of GPS observations whose highest-probability candidate arc is correct (label == 1).
    """
    if df.empty:
        return 0.0

    label_col, group_cols = _resolve_cols(df, label_col, group_cols)

    # Sort descending by probability to stably break ties with first candidate
    sorted_df = df.sort_values(by=prob_col, ascending=False)
    top_candidates = sorted_df.drop_duplicates(subset=group_cols, keep="first")

    acc = float((top_candidates[label_col] == 1).mean())
    return acc


def candidate_miss_rate(
    df: pd.DataFrame,
    label_col: str = None,
    group_cols: List[str] = None
) -> float:
    """
    Computes Candidate Miss Rate:
    Fraction of GPS observations that have NO positive candidate in the candidate set.
    """
    if df.empty:
        return 0.0

    label_col, group_cols = _resolve_cols(df, label_col, group_cols)

    has_positive = df.groupby(group_cols)[label_col].apply(lambda s: bool((s == 1).any()))
    miss_rate = float((~has_positive).mean())
    return miss_rate


def macro_f1(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list]
) -> float:
    """Computes Macro-Averaged F1 Score across binary classes {0, 1}."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_log_loss(
    y_true: Union[np.ndarray, pd.Series, list],
    y_prob: Union[np.ndarray, pd.Series, list],
    eps: float = 1e-15
) -> float:
    """Computes Binary Cross-Entropy (Log Loss) with probability clipping."""
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    return float(sk_log_loss(y_true, y_prob))


def precision(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list]
) -> float:
    """Computes Precision for the positive class (label == 1)."""
    return float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))


def recall(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list]
) -> float:
    """Computes Recall for the positive class (label == 1)."""
    return float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
