import numpy as np
from src.cost_matrix import CLASSES, COST_MATRIX

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}


class CostSensitiveTriageClassifier:
    """
    Inference-ready model wrapping a probabilistic classifier with
    cost-optimal decision decoding under the asymmetric cost matrix.
    """
    def __init__(self, base_estimator, cost_matrix=COST_MATRIX, class_multipliers=None):
        self.base_estimator = base_estimator
        self.cost_matrix = np.array(cost_matrix)
        self.class_multipliers = np.array(class_multipliers) if class_multipliers is not None else np.ones(len(CLASSES))
        self.classes_ = CLASSES

    def fit(self, X, y_idx, sample_weight=None):
        if sample_weight is not None:
            self.base_estimator.fit(X, y_idx, sample_weight=sample_weight)
        else:
            self.base_estimator.fit(X, y_idx)
        return self

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def predict(self, X):
        probs = self.predict_proba(X)
        adjusted_cost = self.cost_matrix * self.class_multipliers
        expected_costs = probs @ adjusted_cost
        pred_idx = np.argmin(expected_costs, axis=1)
        return np.array([IDX_TO_CLASS[i] for i in pred_idx])
