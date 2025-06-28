import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def evaluate_model(model, X_test, y_test, config, model_name="model"):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print(f"\nEvaluation Metrics for {model_name}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # classification report based on test
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n Detailed Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))

    # confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"🔀 Confusion Matrix for {model_name}:\n", cm)

    # Save metrics + report to disk
    output = {
        "model": model_name,
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    save_path = config["output"]["metrics_path"].replace(".json", f"_{model_name}.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n Evaluation report for {model_name} saved to {save_path}")
