# run_demo_v2.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from claim_extraction import extract_claims
from detector_v2 import detect_hallucination
from sklearn.metrics import classification_report, confusion_matrix

with open("test_set.json") as f:
    test_set = json.load(f)

results = []
for ex in test_set:
    claims = extract_claims(ex["output"])
    scores = [detect_hallucination(c, ex["reference"]) for c in claims]
    avg_consistency = sum(scores) / len(scores)
    predicted_label = "not_hallucinated" if avg_consistency > 0.5 else "hallucinated"
    results.append({
        "prompt": ex["prompt"],
        "true_label": ex["label"],
        "predicted_label": predicted_label,
        "hallucination_probability": round(1 - avg_consistency, 3)
    })

y_true = [r["true_label"] for r in results]
y_pred = [r["predicted_label"] for r in results]

print("=== Classification Report (fine-tuned model, on your pharmacy test set) ===")
print(classification_report(y_true, y_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred, labels=["hallucinated", "not_hallucinated"]))

for r in results:
    print(f"\nPrompt: {r['prompt'][:60]}...")
    print(f"  True: {r['true_label']} | Predicted: {r['predicted_label']} | P(hallucinated)={r['hallucination_probability']}")