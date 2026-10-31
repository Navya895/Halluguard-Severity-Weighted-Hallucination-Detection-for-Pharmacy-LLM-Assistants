# build_severity_labeling_set.py
"""
Step 1: Generates a CSV of claims for the team to label independently.
Each team member fills in their own copy's 'label' column, without
looking at anyone else's answers.
"""

import json
import csv
import random

random.seed(42)

# Pull claims from your existing pharmacy-filtered MedHallu pairs
with open("train_val_pairs.json") as f:
    pairs = json.load(f)

all_claims = pairs["train"] + pairs["val"]
random.shuffle(all_claims)

# Take a manageable sample — 50 claims
sample = all_claims[:50]

# Also fold in your own hand-built test set for domain relevance
with open("test_set.json") as f:
    test_set = json.load(f)

for ex in test_set:
    sample.append({"claim": ex["output"], "context": ex["prompt"]})

with open("severity_labeling_sheet.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["claim_id", "claim_text", "context", "label (LOW/MEDIUM/HIGH/CRITICAL — fill this in)"])
    for i, ex in enumerate(sample):
        writer.writerow([i, ex["claim"], ex.get("context", ""), ""])

print(f"Saved severity_labeling_sheet.csv with {len(sample)} claims to label.")
print("Make 3 copies — one per team member — and label independently before comparing.")