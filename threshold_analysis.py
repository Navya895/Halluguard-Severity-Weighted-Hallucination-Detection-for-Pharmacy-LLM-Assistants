import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

MODEL_PATH = "./roberta-halluguard-evidence-v2"
MAX_LENGTH = 512

with open("train_val_pairs_v3.json", encoding="utf-8") as f:
    data = json.load(f)

rows = data["val"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

y_true, p_supported = [], []

for ex in rows:
    inputs = tokenizer(
        ex["context"],
        ex["claim"],
        return_tensors="pt",
        truncation="only_first",
        max_length=MAX_LENGTH
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    y_true.append(int(ex["label"]))
    p_supported.append(float(probs[1]))

y_true = np.array(y_true)
p_supported = np.array(p_supported)

print("Examples:", len(y_true))
print("Current threshold: 0.50")
print("Current mean P(supported):", round(float(p_supported.mean()), 4))
print("\nThreshold | Accuracy | Balanced Acc | Precision | Recall | F1")
print("-" * 68)

for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    pred = (p_supported >= t).astype(int)
    print(
        f"{t:8.2f} | "
        f"{accuracy_score(y_true, pred):8.3f} | "
        f"{balanced_accuracy_score(y_true, pred):12.3f} | "
        f"{precision_score(y_true, pred, zero_division=0):9.3f} | "
        f"{recall_score(y_true, pred, zero_division=0):6.3f} | "
        f"{f1_score(y_true, pred, zero_division=0):.3f}"
    )

print("\nConfusion matrix at threshold 0.50:")
print(confusion_matrix(y_true, (p_supported >= 0.50).astype(int)))
