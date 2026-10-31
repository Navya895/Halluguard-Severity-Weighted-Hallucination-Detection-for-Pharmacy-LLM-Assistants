# train_classifier_v2.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import torch
import joblib
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
encoder = RobertaModel.from_pretrained("roberta-base")
encoder.eval()

def embed(claim, context):
    inputs = tokenizer(context, claim, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = encoder(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze().numpy()

def build_features(subset):
    X, y = [], []
    for i, ex in enumerate(subset):
        X.append(embed(ex["claim"], ex["context"]))
        y.append(ex["label"])
        if (i + 1) % 50 == 0:
            print(f"  embedded {i+1}/{len(subset)}")
    return np.array(X), np.array(y)

with open("train_val_pairs_v2.json") as f:
    data = json.load(f)

print("Embedding training set (v2 — Knowledge-based context)...")
X_train, y_train = build_features(data["train"])
print("Embedding validation set...")
X_val, y_val = build_features(data["val"])

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
print("\n=== Classification Report (v2, Knowledge context) ===")
print(classification_report(y_val, y_pred, target_names=["hallucinated", "not_hallucinated"]))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_val, y_pred))

joblib.dump(clf, "trained_classifier_v2.pkl")
print("\nSaved trained_classifier_v2.pkl")