# run_demo.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import torch
import joblib
from transformers import RobertaTokenizer, RobertaModel
from claim_extraction import extract_claims
from sklearn.metrics import classification_report, confusion_matrix

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
encoder = RobertaModel.from_pretrained("roberta-base")
encoder.eval()
clf = joblib.load("trained_classifier.pkl")

def embed(claim, context):
    inputs = tokenizer(context, claim, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = encoder(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze().numpy()

def detect_hallucination(claim, reference_context):
    emb = embed(claim, reference_context).reshape(1, -1)
    return clf.predict_proba(emb)[0][1]  # prob of "not_hallucinated"

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

print(classification_report(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=["hallucinated", "not_hallucinated"]))

for r in results:
    print(f"\nPrompt: {r['prompt'][:60]}...")
    print(f"  True: {r['true_label']} | Predicted: {r['predicted_label']} | P(hallucinated)={r['hallucination_probability']}")