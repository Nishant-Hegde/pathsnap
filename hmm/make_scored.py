"""Create the classifier-scored candidate tables the HMM needs.

    artifacts/scored/train_oof_scored.pkl
        A's train candidates + `prob` = OUT-OF-FOLD probability from the final
        (tuned) model, using B's 5-fold GroupKFold by trip (models/harness.py).
        Every training row is scored by a model that never saw its trip, so
        HMM parameters can be tuned on train without leakage.
    artifacts/scored/student_test_scored.pkl
        A's student-test candidates + `prob` from the final model
        (models/predict.score_candidates), exactly what the API uses.

Prerequisites:
    python features/build_dataset.py      (A: creates artifacts/features/*_cands.pkl)
    artifacts/model.joblib                (B: tuned model)

Usage:
    python -m hmm.make_scored
"""
from __future__ import annotations

import os
import sys
import time

import joblib
from sklearn.base import clone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.harness import evaluate_model_cv, load_candidate_data, load_feature_cols  # noqa: E402
from models.predict import score_candidates  # noqa: E402

OUT = os.path.join("artifacts", "scored")


def main():
    os.makedirs(OUT, exist_ok=True)
    train_df, test_df = load_candidate_data()
    feature_cols = load_feature_cols()

    t0 = time.time()
    model = clone(joblib.load(os.path.join("artifacts", "model.joblib")))
    cv = evaluate_model_cv(model, train_df, feature_cols, n_splits=5)
    oof = cv["oof_df"]
    oof.to_pickle(os.path.join(OUT, "train_oof_scored.pkl"))
    print(f"train OOF: {len(oof)} rows, CV top-1 {cv['cv_top1_acc_mean']:.4f}, "
          f"{time.time() - t0:.0f}s -> {OUT}/train_oof_scored.pkl")

    st = score_candidates(test_df)
    st.to_pickle(os.path.join(OUT, "student_test_scored.pkl"))
    print(f"student_test: {len(st)} rows -> {OUT}/student_test_scored.pkl")


if __name__ == "__main__":
    main()
