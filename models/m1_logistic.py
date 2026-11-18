"""
Model 1: Logistic Regression (Linear Probabilistic Baseline)
Family: Generalized Linear Models (GLM)
"""
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED
        ))
    ])


def main():
    feature_cols = load_feature_cols()
    train_df, test_df = load_candidate_data(max_train_trips=30)

    pipeline = get_pipeline()
    print("Running 5-Fold GroupKFold CV for Logistic Regression...")
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
        "C": 1.0,
        "solver": "lbfgs",
        "class_weight": "balanced"
    }
    log_experiment("LogisticRegression", "Linear", config, cv_res, test_res)


if __name__ == "__main__":
    main()
