# compare_model_on_testset.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix
from claim_extraction import extract_claims

Q_PATH = "./roberta-halluguard-final"
E_PATH = "./roberta-halluguard-evidence-v2"


def load(path):
    print("Loading model from:", path)
    tok = RobertaTokenizer.from_pretrained(path)
    mdl = RobertaForSequenceClassification.from_pretrained(path)
    mdl.eval()
    return tok, mdl


def p_not_hallucinated(tok, mdl, context, claim, max_len):
    inputs = tok(context, claim, return_tensors="pt",
                 truncation="only_first", max_length=max_len)
    with torch.no_grad():
        logits = mdl(**inputs).logits
    return torch.softmax(logits, dim=1)[0][1].item()


q_tok, q_model = load(Q_PATH)
e_tok, e_model = load(E_PATH)

with open("test_set.json") as f:
    test_set = json.load(f)

y_true, y_q, y_e = [], [], []
print(f"\n{'Prompt':55} {'True':17} {'Q-only':17} {'Evidence-v2':17}")
print("-" * 110)

for ex in test_set:
    claims = extract_claims(ex["output"]) or [ex["output"]]

    q_scores = [p_not_hallucinated(q_tok, q_model, ex["prompt"], c, 256) for c in claims]
    e_ctx = f"{ex['prompt']} {ex['reference']}"
    e_scores = [p_not_hallucinated(e_tok, e_model, e_ctx, c, 512) for c in claims]

    q_pred = "not_hallucinated" if sum(q_scores) / len(q_scores) > 0.5 else "hallucinated"
    e_pred = "not_hallucinated" if sum(e_scores) / len(e_scores) > 0.5 else "hallucinated"

    y_true.append(ex["label"]); y_q.append(q_pred); y_e.append(e_pred)
    print(f"{ex['prompt'][:53]:55} {ex['label']:17} {q_pred:17} {e_pred:17}")

labels = ["hallucinated", "not_hallucinated"]
print("\nQuestion-only model accuracy:", round(accuracy_score(y_true, y_q), 3))
print(confusion_matrix(y_true, y_q, labels=labels))
print("\nEvidence-v2 model accuracy:", round(accuracy_score(y_true, y_e), 3))
print(confusion_matrix(y_true, y_e, labels=labels))