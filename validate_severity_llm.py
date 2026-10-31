# validate_llm_severity.py
import csv
from severity_scorer_llm import compute_severity_llm

with open("severity_labeling_sheet.csv", newline="", encoding="cp1252", errors="replace") as f:
    claims = {row["claim_id"]: row["claim_text"] for row in csv.DictReader(f)}

with open("severity_labels_final.csv", newline="", encoding="utf-8", errors="replace") as f:
    consensus = {row["claim_id"]: row["consensus_label"] for row in csv.DictReader(f)}

correct = 0
total = 0
mismatches = []

for cid, claim_text in claims.items():
    human_label = consensus.get(cid)
    if not human_label:
        continue
    predicted = compute_severity_llm(claim_text)
    total += 1
    if predicted == human_label:
        correct += 1
    else:
        mismatches.append((cid, claim_text[:60], human_label, predicted))
    if total % 10 == 0:
        print(f"  processed {total}/{len(claims)}")

print(f"\nLLM-based scorer matched human consensus on {correct}/{total} ({100*correct/total:.1f}%)")
print("\nMismatches:")
for cid, text, human, pred in mismatches:
    print(f"  [{cid}] '{text}...' — Human: {human}, LLM: {pred}")