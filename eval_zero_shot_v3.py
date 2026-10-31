
"""
HalluGuard: Evidence-Based Detector Evaluation
----------------------------------------------
Evaluates the existing RoBERTa evidence-v2 model.

Expected files:
    roberta-halluguard-evidence-v2/
    pharmacy_train.json
    pharmacy_test.json

Expected JSON format:
[
    {
        "claim": "A factual claim",
        "context": "Question and supporting evidence",
        "label": 1
    }
]

Label convention:
    1 = supported / not hallucinated
    0 = contradicted / hallucinated

This script does NOT train or modify model weights.
The decision threshold is selected on the training split only.
The test split is used only for final evaluation.
"""

import json
from pathlib import Path

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("./training_runs/evidence-v3-20261019-063006-005361/model")
TRAIN_PATH = Path("./pharmacy_train.json")
TEST_PATH = Path("./pharmacy_test.json")

MAX_LENGTH = 512
DEFAULT_THRESHOLD = 0.50


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found: {path.resolve()}\n"
            "Check the filename and run the dataset builder first."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"{path.name} must contain a JSON list of examples."
        )

    if len(data) == 0:
        raise ValueError(
            f"{path.name} is EMPTY. No evaluation can be performed.\n"
            "Check pharmacy_dataset_template.csv and "
            "build_pharmacy_dataset.py."
        )

    required = {"claim", "context", "label"}

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Example {i} in {path.name} is not a JSON object."
            )

        missing = required - set(item.keys())

        if missing:
            raise ValueError(
                f"Example {i} in {path.name} is missing: {missing}"
            )

        if not str(item["claim"]).strip():
            raise ValueError(
                f"Example {i} has an empty claim."
            )

        if not str(item["context"]).strip():
            raise ValueError(
                f"Example {i} has an empty context."
            )

        if item["label"] not in (0, 1):
            raise ValueError(
                f"Example {i} has invalid label: {item['label']}"
            )

    return data


def validate_classes(data, name):
    labels = np.array(
        [int(item["label"]) for item in data],
        dtype=int
    )

    unique, counts = np.unique(labels, return_counts=True)

    print(f"\n{name}: {len(data)} examples")

    for label, count in zip(unique, counts):
        meaning = (
            "Supported"
            if label == 1
            else "Contradicted / hallucinated"
        )
        print(f"  Label {label} ({meaning}): {count}")

    if len(unique) != 2:
        raise ValueError(
            f"{name} must contain both labels 0 and 1."
        )

    return labels


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"\nModel directory not found: "
            f"{MODEL_PATH.resolve()}\n"
            "Update MODEL_PATH to your actual model directory."
        )

    print("\nLoading model:", MODEL_PATH.resolve())

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH)
    )

    # Verify the classifier's label mapping when available.
    id2label = model.config.id2label
    print("Model id2label:", id2label)

    # This script assumes label 1 means supported.
    # Confirm this against your training code.
    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    print("Device:", device)

    return tokenizer, model, device


# ============================================================
# GET P(SUPPORTED) FOR EACH CLAIM
# ============================================================

def score_examples(data, tokenizer, model, device):
    probabilities = []

    with torch.inference_mode():
        for i, item in enumerate(data):
            claim = str(item["claim"])
            context = str(item["context"])

            # RoBERTa pair input:
            # first sequence = evidence/context
            # second sequence = claim
            encoded = tokenizer(
                context,
                claim,
                return_tensors="pt",
                truncation="only_first",
                max_length=MAX_LENGTH,
                padding=False,
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0]

            if len(probs) != 2:
                raise ValueError(
                    "Expected a binary classification model, "
                    f"but received {len(probs)} output classes."
                )

            # Probability of label 1 = supported.
            probabilities.append(
                float(probs[1].item())
            )

            if (i + 1) % 25 == 0:
                print(
                    f"Scored {i + 1}/{len(data)} examples"
                )

    scores = np.asarray(probabilities, dtype=float)

    if len(scores) == 0:
        raise ValueError(
            "No scores were generated. Check the input dataset."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Model produced NaN or infinite scores."
        )

    print(
        "Score range:",
        round(float(scores.min()), 4),
        "to",
        round(float(scores.max()), 4),
    )

    return scores


# ============================================================
# SELECT THRESHOLD USING TRAINING DATA ONLY
# ============================================================

def best_threshold(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if len(scores) == 0 or len(labels) == 0:
        raise ValueError(
            "Cannot select threshold: training scores or labels "
            "are empty. Check the dataset and scoring loop."
        )

    if len(scores) != len(labels):
        raise ValueError(
            "Training scores and labels have different lengths."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Training scores contain invalid values."
        )

    if len(np.unique(labels)) != 2:
        raise ValueError(
            "Threshold selection requires both classes "
            "in the training split."
        )

    # Test the actual score boundaries and midpoints.
    # No empty candidate list and no arbitrary fixed grid.
    unique_scores = np.unique(scores)

    midpoints = (
        (unique_scores[:-1] + unique_scores[1:]) / 2.0
    )

    candidates = np.unique(
        np.concatenate(
            ([0.0, 0.5, 1.0], unique_scores, midpoints)
        )
    )

    best_t = None
    best_metric = -1.0

    for threshold in candidates:
        predictions = (
            scores >= threshold
        ).astype(int)

        metric = balanced_accuracy_score(
            labels, predictions
        )

        if metric > best_metric:
            best_metric = metric
            best_t = float(threshold)

    if best_t is None:
        raise RuntimeError(
            "Threshold selection failed unexpectedly."
        )

    print(
        f"\nSelected threshold: {best_t:.4f}"
    )
    print(
        "Training balanced accuracy at selected threshold:",
        round(best_metric, 4),
    )

    return best_t


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError(
            f"{name}: labels and scores are empty or mismatched."
        )

    predictions = (
        scores >= threshold
    ).astype(int)

    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")
    print(f"Threshold: {threshold:.4f}")

    print(
        "Accuracy:",
        round(accuracy_score(labels, predictions), 4),
    )

    print(
        "Balanced accuracy:",
        round(
            balanced_accuracy_score(labels, predictions),
            4,
        ),
    )

    print(
        "Precision (supported):",
        round(
            precision_score(
                labels, predictions, zero_division=0
            ),
            4,
        ),
    )

    print(
        "Recall (supported):",
        round(
            recall_score(
                labels, predictions, zero_division=0
            ),
            4,
        ),
    )

    print(
        "F1 (supported):",
        round(
            f1_score(
                labels, predictions, zero_division=0
            ),
            4,
        ),
    )

    # AUC measures how well the score ranks positive
    # examples above negative examples.
    print(
        "ROC-AUC:",
        round(roc_auc_score(labels, scores), 4),
    )

    print(
        "\nConfusion matrix "
        "[[true 0 predicted 0/1], [true 1 predicted 0/1]]:"
    )

    print(
        confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=[
                "Hallucinated / contradicted",
                "Supported",
            ],
            zero_division=0,
        )
    )

    return predictions


# ============================================================
# ERROR ANALYSIS
# ============================================================

def print_errors(data, labels, scores, predictions):
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    errors_found = 0

    for i, (item, true_label, score, pred) in enumerate(
        zip(data, labels, scores, predictions)
    ):
        if true_label == pred:
            continue

        errors_found += 1

        if true_label == 0 and pred == 1:
            error_type = (
                "FALSE NEGATIVE FOR HALLUCINATION: "
                "contradicted claim predicted as supported"
            )
        else:
            error_type = (
                "FALSE ALARM: supported claim predicted "
                "as hallucinated"
            )

        print(f"\nExample {i + 1}: {error_type}")
        print(f"True label: {true_label}")
        print(f"Predicted label: {pred}")
        print(f"P(supported): {score:.4f}")
        print("Claim:", item["claim"])
        print("Context:", item["context"][:700])
        print("-" * 60)

    if errors_found == 0:
        print("No classification errors at this threshold.")

    print(f"\nTotal errors: {errors_found}/{len(data)}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("HALLUGUARD DETECTOR EVALUATION")
    print("=" * 60)

    # Load data first, so empty/missing files fail clearly.
    train = load_json(TRAIN_PATH)
    test = load_json(TEST_PATH)

    y_train = validate_classes(train, "Training dataset")
    y_test = validate_classes(test, "Test dataset")

    # Load existing weights; no fine-tuning occurs here.
    tokenizer, model, device = load_model()

    print("\nScoring training examples...")
    train_scores = score_examples(
        train, tokenizer, model, device
    )

    print("\nScoring test examples...")
    test_scores = score_examples(
        test, tokenizer, model, device
    )

    # Threshold selection uses training data only.
    threshold = best_threshold(
        train_scores, y_train
    )

    # Compare the default and training-selected thresholds
    # on the same held-out test set.
    default_predictions = evaluate(
        "TEST RESULTS: DEFAULT THRESHOLD",
        y_test,
        test_scores,
        DEFAULT_THRESHOLD,
    )

    selected_predictions = evaluate(
        "TEST RESULTS: TRAIN-SELECTED THRESHOLD",
        y_test,
        test_scores,
        threshold,
    )

    print_errors(
        test,
        y_test,
        test_scores,
        selected_predictions,
    )

    # Save predictions and scores for later analysis.
    output = []

    for i, item in enumerate(test):
        output.append({
            "claim": item["claim"],
            "context": item["context"],
            "true_label": int(y_test[i]),
            "probability_supported": float(test_scores[i]),
            "prediction_at_0.5": int(default_predictions[i]),
            "prediction_at_selected_threshold": int(
                selected_predictions[i]
            ),
            "threshold": float(threshold),
        })

    output_path = Path("detector_evaluation_results.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(
        f"\nDetailed predictions saved to: "
        f"{output_path.resolve()}"
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
