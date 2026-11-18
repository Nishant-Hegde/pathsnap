"""
Model 4: LightGBM / Gradient Boosting (Sequential Tree Ensemble)
Family: Gradient Boosting (GBDT)
"""
from lightgbm import LGBMClassifier
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
        ("clf", LGBMClassifier(
            learning_rate=0.05,
            n_estimators=100,
            max_depth=6,
            num_leaves=31,
            scale_pos_weight=6.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            verbose=-1
        ))
    ])


def main():
    feature_cols = load_feature_cols()
    train_df, test_df = load_candidate_data(max_train_trips=40)

    pipeline = get_pipeline()
    print("Running 5-Fold GroupKFold CV for LightGBM...")
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
        "learning_rate": 0.05,
        "n_estimators": 100,
        "max_depth": 6,
        "num_leaves": 31,
        "scale_pos_weight": 6.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
    log_experiment("LightGBM", "Gradient Boosting", config, cv_res, test_res)


if __name__ == "__main__":
    main()
