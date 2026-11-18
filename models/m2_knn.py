"""
Model 2: K-Nearest Neighbors (Instance-Based)
Family: Instance-Based / Non-Parametric
"""
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
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
        ("clf", KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            algorithm="kd_tree",
            n_jobs=-1
        ))
    ])


def main():
    feature_cols = load_feature_cols()
    train_df, test_df = load_candidate_data(max_train_trips=5)
    # Stratified subsample per trip for KNN to allow fast neighbor search
    sample_idx = [
        idx for _, grp in train_df.groupby("trip_id")
        for idx in grp.sample(min(len(grp), 2500), random_state=RANDOM_SEED).index
    ]
    train_df = train_df.loc[sample_idx].reset_index(drop=True)
    # Subsample test candidate rows for KNN
    test_sub_trips = test_df["trip_id"].unique()[:2]
    test_sub_df = test_df[test_df["trip_id"].isin(test_sub_trips)].sample(min(5000, len(test_df)), random_state=RANDOM_SEED).copy()

    pipeline = get_pipeline()
    print("Running 5-Fold GroupKFold CV for KNN...")
    cv_res = evaluate_model_cv(pipeline, train_df, feature_cols)
    print(f"CV Macro F1: {cv_res['cv_macro_f1_mean']:.4f} +/- {cv_res['cv_macro_f1_std']:.4f}")
    print(f"CV Top-1 Acc: {cv_res['cv_top1_acc_mean']:.4f}")
    print(f"CV Log Loss: {cv_res['cv_log_loss_mean']:.4f}")

    print("Evaluating on Student-Test set...")
    pipeline.fit(train_df[feature_cols].values, train_df["label"].values)
    test_res = evaluate_student_test(pipeline, test_sub_df, feature_cols)
    print(f"Test Macro F1: {test_res['test_macro_f1']:.4f}")
    print(f"Test Top-1 Acc: {test_res['test_top1_acc']:.4f}")
    print(f"Test Log Loss: {test_res['test_log_loss']:.4f}")
    print(f"Test Inference Time: {test_res['test_infer_time_ms_per_trip']:.2f} ms/trip")

    config = {
        "n_neighbors": 15,
        "weights": "distance",
        "metric": "minkowski",
        "p": 2
    }
    log_experiment("KNN", "Instance-Based", config, cv_res, test_res)


if __name__ == "__main__":
    main()
