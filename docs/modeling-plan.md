# Modeling Plan

This document outlines the proposed preprocessing steps, candidate models, cost matrix integration techniques, and validation strategy based on the PRD requirements and EDA findings.

## 1. Preprocessing Steps

Based on the data report, we must handle significant missingness, class imbalance, and mixed data types. 

* **Imputation Strategy:** 
  * *Reasoning:* With critical vitals like `pain_score`, `dbp`, and `sbp` missing between 20-35% of the time, dropping rows is not viable. Simple mean imputation may distort the distributions of skewed clinical variables. We will use an advanced iterative imputation method (such as `IterativeImputer` or `KNNImputer`) for numeric features to capture the correlated missingness patterns (e.g., utilizing `sbp` to impute `dbp`). Categorical features will use mode imputation or a dedicated "Unknown" category.
* **Categorical Encoding:**
  * *Reasoning:* There are 7 categorical features. Nominal variables (e.g., `race`, `arrival_transport`) will be one-hot encoded to prevent models from inferring false ordinal relationships. The target variable (`start_category`) will be label-encoded (0-3).
* **Feature Scaling:**
  * *Reasoning:* Vital signs are measured on vastly different scales (e.g., temperature vs. sbp). While tree-based models are scale-invariant, linear baseline models (Logistic Regression) require standardized inputs. We will apply `RobustScaler` to numerical variables to mitigate the impact of potential clinical outliers.

## 2. Validation Strategy

* **Stratified K-Fold Cross-Validation (e.g., K=5):**
  * *Reasoning:* The EDA revealed a significant class imbalance (YELLOW: 42.8%, BLACK: 10.4%). A standard random train-test split could lead to folds with disproportionately few examples of the minority classes. Stratified K-Fold ensures that the class distribution is preserved across all training and validation folds, providing a reliable estimate of model performance and preventing overfitting to the majority class.

## 3. Candidate Models Progression

We will iterate from simple, interpretable baselines to advanced, cost-sensitive algorithms:

1. **Baseline Model (Logistic Regression):**
   * *Reasoning:* Establishes a simple, linear benchmark without cost-sensitivity. It provides interpretable coefficients and well-calibrated probabilities out-of-the-box.
2. **Standard Non-linear Ensemble (Random Forest / XGBoost):**
   * *Reasoning:* Tree-based models effectively capture non-linear relationships in clinical vitals and handle the tabular data structure natively. This step establishes our baseline accuracy and F1 scores before introducing cost-awareness.
3. **Cost-Aware Gradient Boosting (LightGBM / XGBoost):**
   * *Reasoning:* The ultimate goal is to minimize the total misclassification cost. We will introduce techniques (outlined below) to heavily penalize dangerous misclassifications (e.g., classifying a RED patient as GREEN).

## 4. Incorporating the Cost Matrix

The PRD explicitly demands that we account for an asymmetric cost matrix where errors have unequal clinical costs. We will employ the following techniques:

* **Class Weighting (During Training):**
  * *Reasoning:* We will map the cost matrix into sample/class weights (e.g., assigning a higher overall weight to RED patients based on the high cost of missing them). This is easily supported via the `class_weight` parameter in most scikit-learn algorithms and nudges the decision boundary away from costly false negatives.
* **Custom Loss Function (During Training):**
  * *Reasoning:* For advanced models like XGBoost or LightGBM, we can define a custom objective function. By calculating the first (gradient) and second (hessian) order derivatives of the asymmetric cost matrix relative to the predictions, the model inherently minimizes the true clinical cost rather than standard log-loss during gradient descent.
* **Threshold Optimization on Predicted Probabilities (Post-Processing):**
  * *Reasoning:* Since the submission requires reporting class probabilities for all four classes, we can decouple probability estimation from the final classification. We will train a standard, well-calibrated probabilistic model. During inference, we multiply the output probability vector by the cost matrix to calculate the *expected cost* of each triage decision. The patient is assigned to the class that yields the minimum expected cost. This mathematically guarantees the optimal cost-sensitive decision for a given probability distribution.

## 5. Phase 3 Baseline Evaluation Findings

Under a 5-Fold Stratified Cross-Validation scheme on `data/processed/train_processed.csv`, three non-cost-sensitive baseline classifiers were benchmarked against the shared evaluation harness (`src/evaluate.py`):

| Model | Accuracy | Macro-F1 | Weighted-F1 | Per-Class F1 (R/Y/G/B) | Total Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 88.25% | 0.8960 | 0.8827 | 0.90 / 0.88 / 0.85 / 0.96 | **226** |
| **XGBoost Classifier** | 87.65% | 0.8889 | 0.8768 | 0.90 / 0.87 / 0.84 / 0.94 | **228** |
| **Random Forest Classifier** | 86.90% | 0.8793 | 0.8688 | 0.87 / 0.87 / 0.85 / 0.94 | **253** |

### Key Takeaways & Transition to Phase 4:
1. **Accuracy vs. Cost Divergence**: Random Forest achieved competitive overall accuracy (86.90%) but suffered the highest misclassification cost (253). This was caused by misclassifying 26 RED patients (20 as YELLOW, 6 as BLACK), which heavily penalized the score under the asymmetric cost matrix.
2. **Standard Loss Blindspot**: Standard cross-entropy and Gini impurity treat every class error equally. In clinical practice, misclassifying an immediate-treatment patient (RED) as minor (GREEN) carries a cost of 10, whereas misclassifying GREEN as YELLOW costs only 1.
3. **Phase 4 Imperative**: Cost-awareness is essential. In Phase 4, we will introduce class weights, custom asymmetric loss functions, and post-processing minimum expected cost thresholding to specifically target and eliminate costly RED and YELLOW under-triage errors.

## 6. Phase 4 Cost-Sensitive Modeling Results & Final Model Selection

In Phase 4, we implemented and evaluated three distinct cost-sensitive approaches under identical 5-Fold Stratified Cross-Validation on `data/processed/train_processed.csv`:

### 1. Distinct Approaches Evaluated

* **Approach 1: Cost-Sensitive Class & Sample Weighting**
  * *Method:* Mapped the asymmetric clinical cost into training weights. For Logistic Regression, class weights were assigned inversely proportional to class frequency and scaled by error severity (`RED: 3.0, YELLOW: 1.3, GREEN: 1.0, BLACK: 1.5`). For XGBoost, sample weights were dynamically assigned during tree construction.
  * *Results:* 
    * `EXP-04 (Cost-Weighted Logistic Reg)`: Total Cost = **227**, RED Recall = 0.896 (up from 0.867).
    * `EXP-05 (Cost-Weighted XGBoost)`: Total Cost = **219**, Accuracy = 87.65%, Macro-F1 = 0.8886.

* **Approach 2: Bayes Expected Cost Decoding (Minimum Expected Loss)**
  * *Method:* Decoupled probability estimation from class decision. Trained a well-calibrated probabilistic gradient boosting model (XGBoost). For each patient, calculated the expected cost vector $\mathbf{e} = \mathbf{p} \cdot \mathbf{C}$ where $\mathbf{C}$ is the official $4 \times 4$ cost matrix. Assigned the triage label via $y^* = \arg\min_k \mathbf{e}_k$.
  * *Results:* 
    * `EXP-06 (XGBoost + Bayes Expected Cost)`: Total Cost = **212** (lowest overall cost), Macro-F1 = 0.8722, RED misclassifications as GREEN = **0**.

* **Approach 3: Post-Processing Cost-Threshold Optimization**
  * *Method:* Conducted a nested cross-validation grid search over class risk multipliers $\mathbf{w}$ to directly optimize clinical thresholds without test leakage ($y^* = \arg\min_k (\mathbf{p} \cdot (\mathbf{C} \odot \mathbf{w}))_k$).
  * *Results:*
    * `EXP-07 (XGBoost + Threshold Optimized)`: Total Cost = **221**, Macro-F1 = 0.8716.

---

### 2. Comprehensive Comparison Table

| Run ID | Model & Method | Approach | Accuracy | CV Macro-F1 | CV Weighted-F1 | Total Cost | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-06** | **XGBoost + Bayes Expected Cost** | **Approach 2 (Bayes Decoding)** | **85.69%** | **0.8722** | **0.8562** | **212** | **SELECTED BEST MODEL** |
| EXP-05 | Cost-Weighted XGBoost | Approach 1 (Sample Weights) | 87.65% | 0.8886 | 0.8766 | 219 | Candidate |
| EXP-07 | XGBoost + Threshold Optimized | Approach 3 (Threshold Tuning) | 85.54% | 0.8716 | 0.8549 | 221 | Candidate |
| EXP-01 | Logistic Regression Baseline | Phase 3 Baseline | 88.25% | 0.8960 | 0.8827 | 226 | Baseline |
| EXP-04 | Cost-Weighted Logistic Reg | Approach 1 (Class Weights) | 85.99% | 0.8732 | 0.8599 | 227 | Candidate |
| EXP-03 | XGBoost Baseline | Phase 3 Baseline | 87.65% | 0.8889 | 0.8768 | 228 | Baseline |
| EXP-02 | Random Forest Baseline | Phase 3 Baseline | 86.90% | 0.8793 | 0.8688 | 253 | Baseline |

---

### 3. Final Model Selection Justification

Per the exit criteria of Phase 4 in `docs/phases.md`:
> *"The final 'Best Model' is selected strictly based on minimizing the total misclassification cost metric."*

1. **Winning Model**: **`EXP-06` (XGBoost with Bayes Expected Cost Decoding)** achieved the minimum Total Misclassification Cost of **212**, outperforming the best Phase 3 baseline by 14 points (and baseline XGBoost by 16 points).
2. **Clinical Safety**: Under Bayes expected cost decoding, **zero** actual RED patients were misclassified as GREEN (the highest cost penalty of 10). RED recall reached 88.1% (119 / 135).
3. **Probability Compliance**: Preserves smooth, well-calibrated probabilities for all four classes as required by the PRD submission format.
4. **Artifact Serialization**: The winning model is wrapped in `CostSensitiveTriageClassifier` from `src.cost_sensitive` and persisted to `models/best_model.pkl` for immediate inference in Phase 5.


