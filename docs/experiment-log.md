# Experiment Log

This document tracks all modeling iterations to ensure the final model is selected based on objective metrics (specifically minimizing Total Misclassification Cost).

| Run ID | Date/Time | Model | Key Hyperparameters | CV Macro-F1 | CV Weighted-F1 | Per-class F1 (R/Y/G/B) | Total Misclassification Cost | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EXP-01 | 2026-09-20 04:25 | Logistic Regression Baseline | max_iter=1000, random_state=42 | 0.896 | 0.883 | 0.90 / 0.88 / 0.85 / 0.96 | 226 | Non-cost-sensitive linear benchmark |
| EXP-02 | 2026-09-20 04:25 | Random Forest Baseline | n_estimators=200, random_state=42, n_jobs=-1 | 0.879 | 0.869 | 0.87 / 0.87 / 0.85 / 0.94 | 253 | Non-cost-sensitive bagged trees benchmark |
| EXP-03 | 2026-09-20 04:25 | XGBoost Baseline | n_estimators=200, lr=0.05, max_depth=4, random_state=42 | 0.889 | 0.877 | 0.90 / 0.87 / 0.84 / 0.94 | 228 | Non-cost-sensitive gradient boosting benchmark |
| EXP-04 | 2026-09-20 04:37 | Cost-Weighted Logistic Reg | class_weight={R:3.0, Y:1.3, G:1.0, B:1.5} | 0.873 | 0.860 | 0.86 / 0.85 / 0.84 / 0.94 | 227 | Approach 1: Cost-matrix derived class weights |
| EXP-05 | 2026-09-20 04:37 | Cost-Weighted XGBoost | sample_weight={R:2.5, Y:1.2, G:1.0, B:1.2} | 0.889 | 0.877 | 0.90 / 0.87 / 0.84 / 0.94 | 219 | Approach 1: Sample weights proportional to clinical cost |
| EXP-06 | 2026-09-20 04:37 | XGBoost + Bayes Expected Cost | argmin(probs @ COST_MATRIX), standard XGBoost | 0.872 | 0.856 | 0.88 / 0.85 / 0.81 / 0.95 | 212 | Approach 2: Bayes optimal decision theory post-processing (Best Model) |
| EXP-07 | 2026-09-20 04:37 | XGBoost + Threshold Optimized | nested CV grid on cost multipliers w | 0.872 | 0.855 | 0.88 / 0.85 / 0.81 / 0.95 | 221 | Approach 3: Post-processing threshold optimization |
