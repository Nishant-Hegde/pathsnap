"""
Bayesian Hyperparameter Optimization for LightGBM on Real Data (Owner: Rishab Naveen)
Uses Optuna with Tree-structured Parzen Estimator (TPE) and fixed seed 42.
Performs GroupKFold CV across training trips to find optimal hyperparameters.
Saves trials.csv, best_params.json, optimization history plot, model.joblib, and model_meta.json.
"""
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

from models.harness import (
    load_feature_cols,
    load_candidate_data,
    evaluate_model_cv,
    evaluate_student_test,
    log_experiment,
    RANDOM_SEED,
    RESULTS_DIR,
    ARTIFACTS_DIR
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, train_df, feature_cols):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 4.0, 8.0),
        "n_estimators": 80,
        "random_state": RANDOM_SEED,
        "verbose": -1
    }

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LGBMClassifier(**params))
    ])

    cv_res = evaluate_model_cv(pipeline, train_df, feature_cols, n_splits=5)
    trial.set_user_attr("cv_top1_acc", cv_res["cv_top1_acc_mean"])
    trial.set_user_attr("cv_log_loss", cv_res["cv_log_loss_mean"])
    return cv_res["cv_macro_f1_mean"]


def plot_optuna_history(study, save_path):
    trials = [t for t in study.trials if t.value is not None]
    trial_nums = [t.number for t in trials]
    values = [t.value for t in trials]
    best_values = [max(values[:i+1]) for i in range(len(values))]

    plt.figure(figsize=(9, 5))
    plt.plot(trial_nums, values, "o", color="#1976D2", alpha=0.6, label="Trial Objective (CV Macro F1)")
    plt.plot(trial_nums, best_values, "-", color="#D32F2F", lw=2.5, label="Best So Far")
    plt.xlabel("Trial Number", fontsize=12)
    plt.ylabel("5-Fold GroupKFold CV Macro F1", fontsize=12)
    plt.title("Bayesian Optimization Progress on Real Data (Optuna TPE)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved optimization history plot to {save_path}")


def main():
    tuning_dir = RESULTS_DIR / "tuning"
    plots_dir = RESULTS_DIR / "plots"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = load_feature_cols()
    train_df, test_df = load_candidate_data(max_train_trips=30)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    n_trials = 25
    print(f"Starting Bayesian Hyperparameter Optimization with Optuna ({n_trials} trials)...")
    study.optimize(lambda trial: objective(trial, train_df, feature_cols), n_trials=n_trials)

    print(f"Optimization finished! Best Trial: #{study.best_trial.number}")
    print(f"Best CV Macro F1: {study.best_value:.4f}")
    best_params = study.best_trial.params

    trials_df = study.trials_dataframe()
    trials_csv = tuning_dir / "trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    print(f"Saved all trial details to {trials_csv}")

    best_params_path = tuning_dir / "best_params.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best parameters to {best_params_path}")

    plot_optuna_history(study, plots_dir / "optuna_tuning_history.png")

    print("\n--- Evaluating Untuned vs Tuned LightGBM ---")
    tuned_clf_params = best_params.copy()
    tuned_clf_params["n_estimators"] = 80
    tuned_clf_params["random_state"] = RANDOM_SEED
    tuned_clf_params["verbose"] = -1

    tuned_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LGBMClassifier(**tuned_clf_params))
    ])

    print("Running 5-Fold GroupKFold CV on Tuned LightGBM...")
    tuned_cv = evaluate_model_cv(tuned_pipeline, train_df, feature_cols)
    tuned_pipeline.fit(train_df[feature_cols].values, train_df["label"].values)
    tuned_test = evaluate_student_test(tuned_pipeline, test_df, feature_cols)

    model_artifact_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(tuned_pipeline, model_artifact_path)
    print(f"Exported final tuned Pipeline to {model_artifact_path}")

    meta = {
        "model_name": "Tuned_LightGBM",
        "family": "Gradient Boosting (GBDT)",
        "feature_cols": feature_cols,
        "best_hyperparameters": best_params,
        "cv_macro_f1": tuned_cv["cv_macro_f1_mean"],
        "cv_top1_acc": tuned_cv["cv_top1_acc_mean"],
        "test_top1_acc": tuned_test["test_top1_acc"],
        "test_log_loss": tuned_test["test_log_loss"],
        "tuning_method": f"Optuna TPE ({n_trials} trials, seed=42)"
    }
    with open(ARTIFACTS_DIR / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log_experiment("Tuned_LightGBM", "Gradient Boosting (Tuned)", best_params, tuned_cv, tuned_test)

    # Export out-of-fold and test predictions for Nishant Hegde's HMM
    features_dir = ARTIFACTS_DIR / "features"
    scored_test_path = features_dir / "scored_student_test.parquet"
    tuned_test["scored_test_df"].to_parquet(scored_test_path, index=False)
    print(f"Exported {scored_test_path} for Nishant Hegde")

    scored_train_oof_path = features_dir / "scored_train_oof.parquet"
    tuned_cv["oof_df"].to_parquet(scored_train_oof_path, index=False)
    print(f"Exported {scored_train_oof_path} for Nishant Hegde")


if __name__ == "__main__":
    main()
