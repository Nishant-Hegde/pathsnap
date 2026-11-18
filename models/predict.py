"""
Candidate Scoring Service (Owner: Rishab Naveen)
Provides score_candidates(cand_df) -> cand_df with extra column 'prob'
Contractually used by Nishant Hegde (HMM/Viterbi) and Prabhu R Habagali (FastAPI inference engine).
"""
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

_CACHED_MODEL: Optional[Pipeline] = None
_CACHED_FEATURE_COLS: Optional[list] = None


def get_model() -> Pipeline:
    """Loads and caches the trained model pipeline."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        model_path = ARTIFACTS_DIR / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. "
                "Run tuning/tune_optuna.py first to produce model.joblib."
            )
        _CACHED_MODEL = joblib.load(model_path)
    return _CACHED_MODEL


def get_feature_cols() -> list:
    """Loads and caches the feature column names."""
    global _CACHED_FEATURE_COLS
    if _CACHED_FEATURE_COLS is None:
        feat_path = ARTIFACTS_DIR / "feature_cols.json"
        with open(feat_path, "r", encoding="utf-8") as f:
            _CACHED_FEATURE_COLS = json.load(f)
    return _CACHED_FEATURE_COLS


def score_candidates(cand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores candidate road segments for each GPS observation.
    Takes a candidate DataFrame from Saanvi Samayam (or API) and adds 'prob' column.
    """
    if cand_df.empty:
        out = cand_df.copy()
        out["prob"] = pd.Series(dtype=float)
        return out

    model = get_model()
    feature_cols = get_feature_cols()

    missing = [c for c in feature_cols if c not in cand_df.columns]
    if missing:
        raise ValueError(f"Candidate DataFrame is missing required feature columns: {missing}")

    X = cand_df[feature_cols].values

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        positive_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    elif hasattr(model, "decision_function"):
        df_vals = model.decision_function(X)
        positive_probs = 1.0 / (1.0 + np.exp(-df_vals))
    else:
        positive_probs = model.predict(X).astype(float)

    out_df = cand_df.copy()
    out_df["prob"] = positive_probs.astype(float)
    return out_df


if __name__ == "__main__":
    feat_dir = ARTIFACTS_DIR / "features"
    test_path = feat_dir / "student_test_cands.pkl"
    if test_path.exists():
        sample_df = pd.read_pickle(test_path).head(15)
        scored = score_candidates(sample_df)
        print("Scoring verification successful! Sample output:")
        print(scored[["trip_id", "point", "arc", "dist", "prob"]].to_string())
