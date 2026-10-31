
# evidence_shuffle_test.py
"""
Evaluate whether the evidence-aware RoBERTa detector uses its context.

A: Correct evidence
B: Mismatched evidence
C: Claim only

Reports accuracy, balanced accuracy, precision, recall, F1,
and confusion matrices. Uses repeated deterministic shuffles.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import random
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

MODEL_PATH = "./roberta-halluguard-evidence-v2"
DATA_PATH = "train_val_pairs_v3.json"
MAX_LENGTH = 512
SEEDS = [21, 42, 73, 101, 202]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
).to(device)
model.eval()

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

val = data["val"]

if not val:
    raise ValueError("Validation dataset is empty.")

labels = np.array([int(ex["label"]) for ex in val])

if set(labels.tolist()) != {0, 1}:
    raise ValueError(
        f"Expected both labels 0 and 1; got {set(labels.tolist())}"
    )

print("Validation examples:", len(val))
print("Label counts:", {
    "unsupported (0)": int(np.sum(labels == 0)),
    "supported (1)": int(np.sum(labels == 1)),
})
print("Model label mapping:", model.config.id2label)


def predict(context, claim):
    encoded = tokenizer(
        context,
        claim,
        return_tensors="pt",
        truncation="only_first",
        max_length=MAX_LENGTH,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]

    predicted = int(torch.argmax(probs).item())
    return predicted, float(probs[1].item())


def evaluate_condition(name, predictions):
    predictions = np.array(predictions, dtype=int)

    print(f"\n{'=' * 60}")
    print(name)
    print("=" * 60)
    print(f"Accuracy:          {accuracy_score(labels, predictions):.3f}")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(labels, predictions):.3f}"
    )
    print(
        f"Precision (class 1): "
        f"{precision_score(labels, predictions, pos_label=1, zero_division=0):.3f}"
    )
    print(
        f"Recall (class 1): "
        f"{recall_score(labels, predictions, pos_label=1, zero_division=0):.3f}"
    )
    print(
        f"F1 (class 1):     "
        f"{f1_score(labels, predictions, pos_label=1, zero_division=0):.3f}"
    )

    print("\nConfusion matrix (rows=true, columns=predicted)")
    print("Order: class 0=unsupported, class 1=supported")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["Unsupported (0)", "Supported (1)"],
            zero_division=0,
        )
    )


# A: Correct evidence
pred_a = [
    predict(ex["context"], ex["claim"])[0]
    for ex in val
]

# B: Repeated mismatched-evidence runs
shuffle_accuracies = []

for seed in SEEDS:
    rng = random.Random(seed)
    contexts = [ex["context"] for ex in val]
    shuffled = contexts[:]

    # Try to create a derangement: no example keeps its own context.
    valid_shuffle = False
    for _ in range(100):
        rng.shuffle(shuffled)
        if all(
            shuffled[i] != contexts[i]
            for i in range(len(contexts))
        ):
            valid_shuffle = True
            break

    if not valid_shuffle:
        print(f"Warning: seed {seed} could not create a full derangement.")
        continue

    predictions = [
        predict(shuffled[i], ex["claim"])[0]
        for i, ex in enumerate(val)
    ]

    shuffle_accuracies.append(accuracy_score(labels, predictions))

    if seed == SEEDS[0]:
        pred_b = predictions

if not shuffle_accuracies:
    raise RuntimeError("Could not create any valid evidence shuffles.")

# C: Claim only
pred_c = [
    predict("", ex["claim"])[0]
    for ex in val
]

evaluate_condition("A) CORRECT EVIDENCE", pred_a)
evaluate_condition("B) MISMATCHED EVIDENCE (first shuffle)", pred_b)
evaluate_condition("C) CLAIM ONLY (no evidence)", pred_c)

print("\n" + "=" * 60)
print("REPEATED SHUFFLE SUMMARY")
print("=" * 60)
print("Seeds evaluated:", len(shuffle_accuracies))
print(
    "Mean mismatched-evidence accuracy:",
    f"{np.mean(shuffle_accuracies):.3f}",
)
print(
    "Std. deviation:",
    f"{np.std(shuffle_accuracies):.3f}",
)
print(
    "Correct-evidence accuracy:",
    f"{accuracy_score(labels, pred_a):.3f}",
)
print(
    "Correct minus mean shuffled accuracy:",
    f"{accuracy_score(labels, pred_a) - np.mean(shuffle_accuracies):.3f}",
)

print(
    "\nInterpretation: a substantial drop after evidence mismatch "
    "suggests sensitivity to context. This test alone does not establish "
    "clinical reliability or generalization to new pharmacy claims."
)
