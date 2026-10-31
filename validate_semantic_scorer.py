# validate_severity_scorer.py
import csv
from severity_scorer import compute_severity

# Reload the same claims used in the labeling sheet
with open("severity_labeling_sheet.csv", newline="", encoding="cp1252", errors="replace") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    claims = {row["claim_id"]: row["claim_text"] for row in rows}

with open("severity_labels_final.csv", newline="", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    consensus = {row["claim_id"]: row["consensus_label"] for row in reader}

correct = 0
total = 0
mismatches = []

for cid, claim_text in claims.items():
    human_label = consensus.get(cid)
    if not human_label:
        continue
    # Using a placeholder hallucination probability of 0.7 since we're
    # testing severity categorization here, not hallucination detection itself
    result = compute_severity(0.7, claim_text)
    predicted = result["tier"]
    total += 1
    if predicted == human_label:
        correct += 1
    else:
        mismatches.append((cid, claim_text[:60], human_label, predicted))

print(f"Rule-based scorer matched human consensus on {correct}/{total} ({100*correct/total:.1f}%)")
print("\nMismatches:")
for cid, text, human, pred in mismatches:
    print(f"  [{cid}] '{text}...' — Human: {human}, Rule-based: {pred}")
