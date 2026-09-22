from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    f1_score, confusion_matrix, classification_report
)
from src.cost_matrix import total_cost, CLASSES

def detailed_metrics(y_true, y_pred):
    results = {}
    results["accuracy"] = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    results["per_class"] = {
        c: {"precision": precision[i], "recall": recall[i], "f1": f1[i], "support": support[i]}
        for i, c in enumerate(CLASSES)
    }

    results["macro_f1"] = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    results["weighted_f1"] = f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=CLASSES)
    results["total_cost"] = total_cost(y_true, y_pred)

    return results

def print_report(results):
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro-F1: {results['macro_f1']:.4f}")
    print(f"Weighted-F1: {results['weighted_f1']:.4f}")
    print(f"Total Misclassification Cost: {results['total_cost']}")
    print("\nPer-class:")
    for c, m in results["per_class"].items():
        print(f"  {c}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])


if __name__ == "__main__":
    y_true = ["RED", "YELLOW", "GREEN", "BLACK", "RED"]
    y_pred = ["YELLOW", "YELLOW", "GREEN", "BLACK", "RED"]
    print_report(detailed_metrics(y_true, y_pred))