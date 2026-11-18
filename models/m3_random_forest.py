"""
Model 3: Random Forest Classifier (Tree Bagging Ensemble)
Family: Ensemble (Bagging)
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from models.harness import (
    load_feature_cols,
    load_candidate_data,
    evaluate_model_cv,
    evaluate_student_test,
    log_experiment,
    RANDOM_SEED
)


def get_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_SEED
        ))
    ])


def main():
    feature_cols = load_feature_cols()
    train_df, test_df = load_candidate_data(max_train_trips=25)

    pipeline = get_pipeline()
    print("Running 5-Fold GroupKFold CV for Random Forest...")
    cv_res = evaluate_model_cv(pipeline, train_df, feature_cols)
    print(f"CV Macro F1: {cv_res['cv_macro_f1_mean']:.4f} +/- {cv_res['cv_macro_f1_std']:.4f}")
    print(f"CV Top-1 Acc: {cv_res['cv_top1_acc_mean']:.4f}")
    print(f"CV Log Loss: {cv_res['cv_log_loss_mean']:.4f}")

    print("Evaluating on Student-Test set...")
    pipeline.fit(train_df[feature_cols].values, train_df["label"].values)
    test_res = evaluate_student_test(pipeline, test_df, feature_cols)
    print(f"Test Macro F1: {test_res['test_macro_f1']:.4f}")
    print(f"Test Top-1 Acc: {test_res['test_top1_acc']:.4f}")
    print(f"Test Log Loss: {test_res['test_log_loss']:.4f}")
    print(f"Test Inference Time: {test_res['test_infer_time_ms_per_trip']:.2f} ms/trip")

    config = {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "class_weight": "balanced_subsample"
    }
    log_experiment("RandomForest", "Bagging Ensemble", config, cv_res, test_res)


if __name__ == "__main__":
    main()
