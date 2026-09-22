# Project Phases and Timeline

This document breaks down the Healthcare Triage Classification (MM26ML03) project into sequential phases, outlining the tasks, assigning ownership based on the team playbook, estimating time, and defining strict exit criteria for each phase.

---

## Phase 1: Data Understanding & Planning
* **Owner:** Sudeep Kumar Sahu
* **Tasks:**
  * Perform Exploratory Data Analysis (EDA) on the raw datasets.
  * Generate the `data-report.md` summarizing class balance, missing values, and data types.
  * Finalize all 6 planning documents in the `docs/` folder (PRD, rules, modeling-plan, phases, etc.).
* **Exit Criteria:**
  * EDA notebook (`01_eda.ipynb`) executes without errors.
  * `docs/data-report.md` and `docs/modeling-plan.md` are complete and reflect the current data state.

## Phase 2: Preprocessing & Evaluation Harness
* **Owner:** Sudeep Kumar Sahu
* **Tasks:**
  * Build the robust preprocessing pipeline (`src/preprocessing.py`) handling iterative imputation, robust scaling, and categorical encoding.
  * Implement the shared metric logic (`src/evaluate.py`) and hardcode the explicit asymmetric cost matrix (`src/cost_matrix.py`).
  * Process the raw data and export to `data/processed/train_clean.csv` and `test_clean.csv`.
* **Exit Criteria:**
  * Clean, fully numeric, and scaled datasets are successfully saved in `data/processed/`.
  * The evaluation harness runs successfully on dummy predictions and outputs all required metrics (including total misclassification cost).

## Phase 3: Baseline Modeling 
* **Owner:** Shrijay Sinha (Modeling Owner)
* **Tasks:**
  * Train an initial, non-cost-sensitive baseline model (Logistic Regression).
  * Train a standard tree-based ensemble (Random Forest / XGBoost) for baseline accuracy.
  * Run predictions through the shared evaluation harness.
* **Exit Criteria:**
  * Baseline metrics (Accuracy, F1-scores, standard Confusion Matrix, and baseline Total Cost) are recorded in `docs/experiment-log.md`.

## Phase 4: Cost-Sensitive Modeling Iterations
* **Owner:** Shrijay Sinha (Modeling Owner)
* **Tasks:**
  * Implement cost-sensitive techniques to directly penalize critical misclassifications (e.g., RED misclassified as GREEN).
  * Iterate using Class Weights, Custom Objective/Loss Functions (for XGBoost/LightGBM), and Post-processing Threshold Optimization.
  * Tune hyperparameters and log every run.
* **Exit Criteria:**
  * At least 3 distinct cost-sensitive approaches are evaluated and logged in `docs/experiment-log.md`.
  * The final "Best Model" is selected strictly based on minimizing the total misclassification cost metric.

## Phase 5: Submission & Documentation
* **Owner:** Sudeep Kumar Sahu 
* **Tasks:**
  * Run the final selected model in `notebooks/04_final_predictions.ipynb`.
  * Format the output CSV exactly as required (`outputs/{team_ID}_MM26ML03.csv`) preserving patient ordering and the 4 probability columns.
  * Draft the final 1-page `README.md`.
  * Generate presentation visualizations (Confusion Matrix Heatmap, Feature Importance plot).
  * Run through the `docs/rules.md` checklist.
* **Exit Criteria:**
  * All checkboxes in `docs/rules.md` are checked.
  * The final submission CSV, README, and Notebooks are packaged and ready for GitHub/ZIP upload.
