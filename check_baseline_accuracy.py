# check_baseline_accuracy.py
"""
Re-evaluates the saved question-only model (roberta-halluguard-final) on the
validation split in train_val_pairs.json, using the original settings.

Expected if nothing has changed: 94 examples, accuracy 0.947, confusion [[43 2] [3 46]].
Also prints library versions and file timestamps so we can tell whether the model
or the data file was overwritten, or whether the environment changed.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import datetime
import json
import numpy as np
import torch
import transformers
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix

MODEL = "./roberta-halluguard-final"
DATA = "train_val_pairs.json"


def stamp(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")


print("torch", torch.__version__, "| transformers", transformers.__version__)
for name in sorted(os.listdir(MODEL)):
    if name.endswith((".safetensors", ".bin")):
        print(f"model weights {name} modified: {stamp(os.path.join(MODEL, name))}")
print(f"{DATA} modified: {stamp(DATA)}")

tok = RobertaTokenizer.from_pretrained(MODEL)
model = RobertaForSequenceClassification.from_pretrained(MODEL)
model.eval()

with open(DATA) as f:
    val = json.load(f)["val"]

preds, labels = [], []
for ex in val:
    inputs = tok(ex["context"], ex["claim"], return_tensors="pt",
                 truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    preds.append(int(torch.argmax(logits, dim=1).item()))
    labels.append(ex["label"])

print(f"\nValidation examples: {len(val)}")
print(f"Accuracy: {accuracy_score(labels, preds):.3f}")
print(confusion_matrix(labels, preds))
print("\nExpected if unchanged: 94 examples, accuracy 0.947, [[43 2] [3 46]]")