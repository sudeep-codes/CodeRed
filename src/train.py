import sys
import os
from pathlib import Path
from datetime import datetime

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.cost_matrix import CLASSES, COST_MATRIX, total_cost
from src.evaluate import detailed_metrics, print_report
from src.cost_sensitive import CostSensitiveTriageClassifier

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# Average misclassification cost per true class from COST_MATRIX:
# RED: (5 + 10 + 5) / 3 = 6.67
# YELLOW: (2 + 3 + 2) / 3 = 2.33
# GREEN: (1 + 1 + 1) / 3 = 1.00
# BLACK: (2 + 2 + 2) / 3 = 2.00
COST_CLASS_WEIGHTS = {
    CLASS_TO_IDX["RED"]: 3.0,
    CLASS_TO_IDX["YELLOW"]: 1.3,
    CLASS_TO_IDX["GREEN"]: 1.0,
    CLASS_TO_IDX["BLACK"]: 1.5,
}


def load_data(filepath="data/processed/train_processed.csv"):
    df = pd.read_csv(filepath)
    X = df.drop(columns=["target"])
    y = df["target"].values
    return X, y


def cross_validate_model(model_factory, X, y, cv=5, random_state=42, model_name="Model",
                         sample_weight_fn=None, decode_mode="argmax", optimize_multipliers=False):
    """
    Performs Stratified 5-Fold Cross-Validation.
    Supports:
    - Standard argmax decoding ('argmax')
    - Bayes Expected Cost Decoding ('bayes')
    - Nested Threshold / Cost-Multiplier Optimization ('nested_threshold')
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    y_idx = np.array([CLASS_TO_IDX[label] for label in y])

    oof_preds = np.empty(len(y), dtype=object)
    oof_probs = np.zeros((len(y), len(CLASSES)))
    fold_metrics = []
    best_multipliers_list = []

    print(f"\n{'='*60}")
    print(f"Running {cv}-Fold Stratified CV for: {model_name}")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_idx), 1):
        X_train, y_train_idx = X.iloc[train_idx], y_idx[train_idx]
        X_val, y_val_idx = X.iloc[val_idx], y_idx[val_idx]
        y_val_str = y[val_idx]
        y_train_str = y[train_idx]

        model = model_factory()

        if sample_weight_fn is not None:
            sw_train = np.array([sample_weight_fn(c) for c in y_train_idx])
            model.fit(X_train, y_train_idx, sample_weight=sw_train)
        else:
            model.fit(X_train, y_train_idx)

        val_probs = model.predict_proba(X_val)
        train_probs = model.predict_proba(X_train)
        oof_probs[val_idx] = val_probs

        if decode_mode == "argmax":
            val_pred_idx = np.argmax(val_probs, axis=1)
        elif decode_mode == "bayes":
            expected_costs = val_probs @ COST_MATRIX
            val_pred_idx = np.argmin(expected_costs, axis=1)
        elif decode_mode == "nested_threshold":
            # Search best multipliers w on training fold to minimize total cost
            best_c = float("inf")
            best_w = np.ones(len(CLASSES))
            for red_w in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
                for yel_w in [0.8, 0.9, 1.0, 1.1, 1.2]:
                    for grn_w in [0.8, 0.9, 1.0, 1.1, 1.2]:
                        w = np.array([red_w, yel_w, grn_w, 1.0])
                        pred_train_idx = np.argmin(train_probs @ (COST_MATRIX * w), axis=1)
                        pred_train_str = [IDX_TO_CLASS[k] for k in pred_train_idx]
                        c = total_cost(y_train_str, pred_train_str)
                        if c < best_c:
                            best_c = c
                            best_w = w.copy()

            best_multipliers_list.append(best_w)
            val_pred_idx = np.argmin(val_probs @ (COST_MATRIX * best_w), axis=1)
        else:
            raise ValueError(f"Unknown decode_mode: {decode_mode}")

        val_preds_str = np.array([IDX_TO_CLASS[i] for i in val_pred_idx])
        oof_preds[val_idx] = val_preds_str

        fold_res = detailed_metrics(y_val_str, val_preds_str)
        fold_metrics.append(fold_res)
        print(f"  Fold {fold} - Acc: {fold_res['accuracy']:.4f} | Macro-F1: {fold_res['macro_f1']:.4f} | Weighted-F1: {fold_res['weighted_f1']:.4f} | Cost: {fold_res['total_cost']}")

    oof_metrics = detailed_metrics(y, oof_preds)
    print(f"\n--- OOF Aggregate Results for {model_name} ---")
    print_report(oof_metrics)

    mean_cost = np.mean([m["total_cost"] for m in fold_metrics])
    std_cost = np.std([m["total_cost"] for m in fold_metrics])
    mean_macro = np.mean([m["macro_f1"] for m in fold_metrics])
    std_macro = np.std([m["macro_f1"] for m in fold_metrics])
    mean_weighted = np.mean([m["weighted_f1"] for m in fold_metrics])
    std_weighted = np.std([m["weighted_f1"] for m in fold_metrics])

    print(f"CV Macro-F1 (mean +/- std): {mean_macro:.4f} +/- {std_macro:.4f}")
    print(f"CV Weighted-F1 (mean +/- std): {mean_weighted:.4f} +/- {std_weighted:.4f}")
    print(f"CV Total Cost (sum = {oof_metrics['total_cost']}, fold mean +/- std = {mean_cost:.1f} +/- {std_cost:.1f})")

    avg_multipliers = np.mean(best_multipliers_list, axis=0) if best_multipliers_list else np.ones(len(CLASSES))

    return {
        "model_name": model_name,
        "oof_metrics": oof_metrics,
        "oof_preds": oof_preds,
        "oof_probs": oof_probs,
        "fold_metrics": fold_metrics,
        "mean_macro": mean_macro,
        "std_macro": std_macro,
        "mean_weighted": mean_weighted,
        "std_weighted": std_weighted,
        "total_cost": oof_metrics["total_cost"],
        "avg_multipliers": avg_multipliers,
    }


def format_per_class_f1(per_class_dict):
    r_f1 = per_class_dict.get("RED", {}).get("f1", 0.0)
    y_f1 = per_class_dict.get("YELLOW", {}).get("f1", 0.0)
    g_f1 = per_class_dict.get("GREEN", {}).get("f1", 0.0)
    b_f1 = per_class_dict.get("BLACK", {}).get("f1", 0.0)
    return f"{r_f1:.2f} / {y_f1:.2f} / {g_f1:.2f} / {b_f1:.2f}"


def update_experiment_log(results_list, log_path="docs/experiment-log.md"):
    log_file = Path(log_path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    header = (
        "# Experiment Log\n\n"
        "This document tracks all modeling iterations to ensure the final model is selected based on objective metrics (specifically minimizing Total Misclassification Cost).\n\n"
        "| Run ID | Date/Time | Model | Key Hyperparameters | CV Macro-F1 | CV Weighted-F1 | Per-class F1 (R/Y/G/B) | Total Misclassification Cost | Notes |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )

    rows = []
    for res in results_list:
        run_id = res['run_id']
        model = res['model_name']
        params = res['hyperparameters']
        macro_f1 = f"{res['oof_metrics']['macro_f1']:.3f}"
        weighted_f1 = f"{res['oof_metrics']['weighted_f1']:.3f}"
        per_class = format_per_class_f1(res['oof_metrics']['per_class'])
        cost = f"{res['total_cost']}"
        notes = res['notes']
        row = f"| {run_id} | {now_str} | {model} | {params} | {macro_f1} | {weighted_f1} | {per_class} | {cost} | {notes} |"
        rows.append(row)

    new_content = header + "\n".join(rows) + "\n"
    log_file.write_text(new_content, encoding="utf-8")
    print(f"\nSuccessfully updated {log_path} with {len(rows)} experiments.")


def run_all_experiments():
    X, y = load_data("data/processed/train_processed.csv")
    y_idx = np.array([CLASS_TO_IDX[label] for label in y])
    print(f"Loaded training data: X={X.shape}, y={y.shape}")

    experiments = [
        # --- PHASE 3 BASELINES ---
        {
            "run_id": "EXP-01",
            "model_name": "Logistic Regression Baseline",
            "hyperparameters": "max_iter=1000, random_state=42",
            "notes": "Non-cost-sensitive linear benchmark",
            "factory": lambda: LogisticRegression(max_iter=1000, random_state=42),
            "sample_weight_fn": None,
            "decode_mode": "argmax",
        },
        {
            "run_id": "EXP-02",
            "model_name": "Random Forest Baseline",
            "hyperparameters": "n_estimators=200, random_state=42, n_jobs=-1",
            "notes": "Non-cost-sensitive bagged trees benchmark",
            "factory": lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "sample_weight_fn": None,
            "decode_mode": "argmax",
        },
        {
            "run_id": "EXP-03",
            "model_name": "XGBoost Baseline",
            "hyperparameters": "n_estimators=200, lr=0.05, max_depth=4, random_state=42",
            "notes": "Non-cost-sensitive gradient boosting",
            "factory": lambda: XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42,
                eval_metric="mlogloss", n_jobs=-1
            ),
            "sample_weight_fn": None,
            "decode_mode": "argmax",
        },

        # --- PHASE 4 COST-SENSITIVE APPROACHES ---
        # Approach 1: Class / Sample Weighting
        {
            "run_id": "EXP-04",
            "model_name": "Cost-Weighted Logistic Reg",
            "hyperparameters": "class_weight={R:3.0, Y:1.3, G:1.0, B:1.5}",
            "notes": "Approach 1: Cost-matrix derived class weights",
            "factory": lambda: LogisticRegression(max_iter=1000, random_state=42, class_weight=COST_CLASS_WEIGHTS),
            "sample_weight_fn": None,
            "decode_mode": "argmax",
        },
        {
            "run_id": "EXP-05",
            "model_name": "Cost-Weighted XGBoost",
            "hyperparameters": "sample_weight={R:2.5, Y:1.2, G:1.0, B:1.2}",
            "notes": "Approach 1: Sample weights proportional to clinical cost",
            "factory": lambda: XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42,
                eval_metric="mlogloss", n_jobs=-1
            ),
            "sample_weight_fn": lambda c: {0: 2.5, 1: 1.2, 2: 1.0, 3: 1.2}.get(c, 1.0),
            "decode_mode": "argmax",
        },

        # Approach 2: Bayes Expected Cost Decoding
        {
            "run_id": "EXP-06",
            "model_name": "XGBoost + Bayes Expected Cost",
            "hyperparameters": "argmin(probs @ COST_MATRIX), standard XGBoost",
            "notes": "Approach 2: Bayes optimal decision theory post-processing",
            "factory": lambda: XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42,
                eval_metric="mlogloss", n_jobs=-1
            ),
            "sample_weight_fn": None,
            "decode_mode": "bayes",
        },

        # Approach 3: Post-Processing Cost-Threshold Optimization
        {
            "run_id": "EXP-07",
            "model_name": "XGBoost + Threshold Optimized",
            "hyperparameters": "nested CV grid on cost multipliers w",
            "notes": "Approach 3: Post-processing threshold optimization",
            "factory": lambda: XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42,
                eval_metric="mlogloss", n_jobs=-1
            ),
            "sample_weight_fn": None,
            "decode_mode": "nested_threshold",
        },
    ]

    all_results = []
    for exp in experiments:
        res = cross_validate_model(
            model_factory=exp["factory"],
            X=X,
            y=y,
            cv=5,
            random_state=42,
            model_name=exp["model_name"],
            sample_weight_fn=exp["sample_weight_fn"],
            decode_mode=exp["decode_mode"],
        )
        res["run_id"] = exp["run_id"]
        res["hyperparameters"] = exp["hyperparameters"]
        res["notes"] = exp["notes"]
        res["factory"] = exp["factory"]
        res["decode_mode"] = exp["decode_mode"]
        res["sample_weight_fn"] = exp["sample_weight_fn"]
        all_results.append(res)

    update_experiment_log(all_results, "docs/experiment-log.md")

    # Select winning Best Model strictly based on minimizing Total Cost
    best_exp = min(all_results, key=lambda r: r["total_cost"])
    print(f"\n{'*'*60}")
    print(f"WINNING BEST MODEL: {best_exp['model_name']} ({best_exp['run_id']})")
    print(f"Total Misclassification Cost: {best_exp['total_cost']}")
    print(f"CV Macro-F1: {best_exp['oof_metrics']['macro_f1']:.4f}")
    print(f"CV Weighted-F1: {best_exp['oof_metrics']['weighted_f1']:.4f}")
    print(f"{'*'*60}\n")

    # Train final model on FULL training set
    print("Training winning model on full dataset and exporting to models/best_model.pkl ...")
    full_model = best_exp["factory"]()
    if best_exp["sample_weight_fn"] is not None:
        sw_full = np.array([best_exp["sample_weight_fn"](c) for c in y_idx])
        full_model.fit(X, y_idx, sample_weight=sw_full)
    else:
        full_model.fit(X, y_idx)

    best_multipliers = best_exp["avg_multipliers"]
    best_pipeline = CostSensitiveTriageClassifier(
        base_estimator=full_model,
        cost_matrix=COST_MATRIX,
        class_multipliers=best_multipliers if best_exp["decode_mode"] == "nested_threshold" else np.ones(len(CLASSES))
    )

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "best_model.pkl"
    joblib.dump(best_pipeline, model_path)
    print(f"Successfully serialized final model to {model_path}!")

    return all_results, best_pipeline


if __name__ == "__main__":
    run_all_experiments()
