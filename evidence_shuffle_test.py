# evidence_shuffle_test.py
"""
Does the evidence-grounded model actually USE the evidence?

Inference only (no training). On the same 94 validation claims, score the model with:
  A) the correct evidence          (what it was trained on)
  B) evidence from a DIFFERENT example (mismatched, shuffled with a fixed seed)
  C) no context at all (claim only)

If accuracy barely drops in B and C, the model is judging the claim's wording,
not checking it against the evidence.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import random
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score
PATH = "./roberta-halluguard-evidence-v2"
tok = RobertaTokenizer.from_pretrained(PATH)
model = RobertaForSequenceClassification.from_pretrained(PATH)
model.eval()

with open("train_val_pairs_v3.json") as f:
    val = json.load(f)["val"]


def predict(context, claim):
    inputs = tok(context, claim, return_tensors="pt",
                 truncation="only_first", max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    return int(torch.argmax(logits, dim=1).item())   # 1 = not hallucinated


labels = [ex["label"] for ex in val]

# A) correct evidence
pred_a = [predict(ex["context"], ex["claim"]) for ex in val]

# B) mismatched evidence (each claim gets another example's context)
random.seed(42)
contexts = [ex["context"] for ex in val]
shuffled = contexts[:]
while any(s == c for s, c in zip(shuffled, contexts)):
    random.shuffle(shuffled)
pred_b = [predict(shuffled[i], ex["claim"]) for i, ex in enumerate(val)]

# C) claim only
pred_c = [predict("", ex["claim"]) for ex in val]

print(f"A) correct evidence   : accuracy {accuracy_score(labels, pred_a):.3f}")
print(f"B) mismatched evidence: accuracy {accuracy_score(labels, pred_b):.3f}")
print(f"C) claim only (no ctx): accuracy {accuracy_score(labels, pred_c):.3f}")
print("\nIf B and C stay close to A, the model is not really using the evidence.")