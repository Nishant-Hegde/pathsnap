"""
Model 5: Multi-Layer Perceptron (Neural Network)
Family: Neural Networks / Deep Learning
Saves genuine epoch-wise loss curve to results/plots/mlp_loss_curve.png
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.harness import (
    load_feature_cols,
    load_candidate_data,
    evaluate_model_cv,
    evaluate_student_test,
    log_experiment,
    RANDOM_SEED,
    RESULTS_DIR
)


def get_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=512,
            learning_rate_init=0.005,
            max_iter=50,
            early_stopping=True,
            n_iter_no_change=8,
            validation_fraction=0.15,
            random_state=RANDOM_SEED
        ))
    ])


def plot_mlp_loss_curve(mlp_model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    epochs = range(1, len(mlp_model.loss_curve_) + 1)
    ax1.plot(epochs, mlp_model.loss_curve_, "b-", lw=2, label="Train Loss (Binary Cross-Entropy)")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Training Loss", color="b", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, alpha=0.3)

    if hasattr(mlp_model, "validation_scores_") and mlp_model.validation_scores_:
        ax2 = ax1.twinx()
        ax2.plot(epochs, mlp_model.validation_scores_, "r--", lw=2, label="Validation Accuracy")
        ax2.set_ylabel("Validation Score", color="r", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="r")

    plt.title("MLP Epoch-Wise Training Loss & Validation Curve (Real Data)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved genuine MLP epoch loss curve to {save_path}")


def main():
    feature_cols = load_feature_cols()
    train_df, test_df = load_candidate_data(max_train_trips=8)

    pipeline = get_pipeline()
    print("Running 5-Fold GroupKFold CV for MLP...")
    cv_res = evaluate_model_cv(pipeline, train_df, feature_cols)
    print(f"CV Macro F1: {cv_res['cv_macro_f1_mean']:.4f} +/- {cv_res['cv_macro_f1_std']:.4f}")
    print(f"CV Top-1 Acc: {cv_res['cv_top1_acc_mean']:.4f}")
    print(f"CV Log Loss: {cv_res['cv_log_loss_mean']:.4f}")

    print("Evaluating on Student-Test set & generating loss curve...")
    pipeline.fit(train_df[feature_cols].values, train_df["label"].values)
    
    plots_dir = RESULTS_DIR / "plots"
    mlp_clf = pipeline.named_steps["clf"]
    plot_mlp_loss_curve(mlp_clf, plots_dir / "mlp_loss_curve.png")

    test_res = evaluate_student_test(pipeline, test_df, feature_cols)
    print(f"Test Macro F1: {test_res['test_macro_f1']:.4f}")
    print(f"Test Top-1 Acc: {test_res['test_top1_acc']:.4f}")
    print(f"Test Log Loss: {test_res['test_log_loss']:.4f}")
    print(f"Test Inference Time: {test_res['test_infer_time_ms_per_trip']:.2f} ms/trip")

    config = {
        "hidden_layer_sizes": [64, 32],
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "learning_rate_init": 0.005,
        "max_iter": 100
    }
    log_experiment("MLP", "Neural Network", config, cv_res, test_res)


if __name__ == "__main__":
    main()
