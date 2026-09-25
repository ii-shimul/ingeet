import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from src.config import (
    DEFAULT_CLASSES_PATH,
    DEFAULT_MODEL_PATH,
    PROCESSED_DATA_DIR,
)


def evaluate_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    data_dir: Path = PROCESSED_DATA_DIR,
    classes_path: Path = DEFAULT_CLASSES_PATH,
    output_dir: Path = Path("reports"),
):
    """Evaluate trained LSTM model against test split with full error analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {model_path}")
    model = load_model(str(model_path))

    x_test_path = data_dir / "X_test.npy"
    y_test_path = data_dir / "y_test.npy"

    assert x_test_path.exists(), f"Missing test features: {x_test_path}"
    assert y_test_path.exists(), f"Missing test labels: {y_test_path}"

    X_test = np.load(str(x_test_path))
    y_test = np.load(str(y_test_path))
    class_names = (
        np.load(str(classes_path), allow_pickle=True)
        if Path(classes_path).exists()
        else [str(i) for i in range(len(np.unique(y_test)))]
    )

    print(f"Test samples: {X_test.shape[0]}, Features per frame: {X_test.shape[2]}")

    # Standard evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nOverall Test Loss: {test_loss:.4f}")
    print(f"Overall Test Accuracy: {test_acc * 100:.2f}%")

    # Predictions
    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=[str(c) for c in class_names],
        digits=4,
    )
    print("\nClassification Report:\n", report)

    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report)
    print(f"Saved report to: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(20, 18))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (60 Bengali Sign Classes)")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # Identify top 5 most confused sign classes
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    weak_indices = np.argsort(per_class_acc)[:5]
    print("\nTop 5 Most Confused Classes (Lowest Recall):")
    for idx in weak_indices:
        print(f"- {class_names[idx]}: {per_class_acc[idx] * 100:.1f}% accuracy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Ingeet LSTM Model")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to .keras model")
    parser.add_argument("--data", type=str, default=str(PROCESSED_DATA_DIR), help="Path to processed data dir")
    parser.add_argument("--classes", type=str, default=str(DEFAULT_CLASSES_PATH), help="Path to class_names.npy")
    parser.add_argument("--output", type=str, default="reports", help="Output directory for reports")
    args = parser.parse_args()

    evaluate_model(
        model_path=Path(args.model),
        data_dir=Path(args.data),
        classes_path=Path(args.classes),
        output_dir=Path(args.output),
    )
