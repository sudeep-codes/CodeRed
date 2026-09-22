# Healthcare Triage Classification (MM26ML03)
**Team ID:** MM2645

## Problem Statement
In emergency mass-casualty scenarios, accurately prioritizing patients (triage) using the START protocol is extremely difficult. Misclassifying a critical patient as stable comes with a severe clinical penalty, costing lives. Our goal was to automate triage assignment using patient vitals, consciousness scores, and demographics while explicitly minimizing a strict, asymmetric cost matrix where certain false negatives (e.g., misclassifying a RED patient as GREEN) carry high penalties.

## Approach & Methodology
We implemented a robust machine learning pipeline tailored for cost-sensitive classification:

1. **Preprocessing**: Missing values were handled effectively, categorical variables encoded, and numerical features scaled robustly to handle outliers typical of emergency clinical data.
2. **Modeling Pipeline**: We benchmarked several baseline models (Logistic Regression, Random Forest, XGBoost) and found XGBoost to be the best performing base estimator.
3. **Cost-Sensitive Decoding**: Rather than relying on simple class weights, we used **Bayes Optimal Expected Cost Decoding**. The base XGBoost model predicts the raw probabilities for the 4 classes. During inference, we multiply the predicted probabilities by the given cost matrix. The class that minimizes the expected clinical cost is then chosen as the final prediction.

## Final Model & Results
Our best model (**XGBoost + Bayes Expected Cost Post-processing**) was strictly selected based on its ability to minimize the Total Misclassification Cost on our cross-validation splits.

**Key Metrics (from Cross-Validation):**
* **Macro-F1:** ~0.872
* **Weighted-F1:** ~0.856
* **Total Misclassification Cost:** 212 (best observed across all experiments)

This cost-optimal decision theory approach effectively forces the model to be highly sensitive to critical cases (RED), avoiding the severe penalties associated with under-triaging patients.

## Deliverables
- `outputs/MM2645_MM26ML03.csv`: Final test predictions containing patient IDs, assigned triage categories, and the 4 probability columns.
- `notebooks/04_final_predictions.ipynb`: Inference notebook containing the final prediction execution.
- `outputs/confusion_matrix.png` & `outputs/feature_importance.png`: Visualizations of the final model's performance on the training data.
