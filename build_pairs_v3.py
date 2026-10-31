# build_pairs_v3.py
"""
Builds evidence-grounded training pairs (Question + Knowledge as context)
while KEEPING THE EXACT SAME train/val split as train_val_pairs.json.

Why: the existing 94.7% model was trained/evaluated on train_val_pairs.json
(question-only context). To compare fairly, the new model must be evaluated
on the same validation claims. So instead of reshuffling, we look up the
evidence passage for every claim already in that file.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from datasets import load_dataset

ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")["train"]


def knowledge_text(example):
    k = example["Knowledge"]
    return " ".join(k) if isinstance(k, list) else k


# Map each claim text -> (question, evidence passage)
lookup = {}
for ex in ds:
    ev = knowledge_text(ex)
    lookup[ex["Ground Truth"].strip()] = (ex["Question"], ev)
    lookup[ex["Hallucinated Answer"].strip()] = (ex["Question"], ev)

with open("train_val_pairs.json") as f:
    old = json.load(f)


def convert(split_name):
    out, missing = [], 0
    for ex in old[split_name]:
        hit = lookup.get(ex["claim"].strip())
        if hit is None:
            missing += 1
            continue
        question, evidence = hit
        out.append({
            "claim": ex["claim"],
            "context": f"{question} {evidence}",
            "label": ex["label"],
        })
    print(f"{split_name}: kept {len(out)} / {len(old[split_name])} (unmatched: {missing})")
    return out


new = {"train": convert("train"), "val": convert("val")}

with open("train_val_pairs_v3.json", "w") as f:
    json.dump(new, f)

print("Saved train_val_pairs_v3.json (same split as the 94.7% model, evidence-grounded context)")