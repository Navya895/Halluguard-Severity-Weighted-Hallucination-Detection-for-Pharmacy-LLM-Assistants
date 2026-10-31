# merge_severity_labels.py
"""
Step 2: After all 3 team members have independently filled in the
'label' column of their own copy of severity_labeling_sheet.csv,
run this to merge them, compute agreement, and produce a final
consensus-labeled dataset.

Expects 3 files: labels_navya.csv, labels_reshma.csv, labels_vedika.csv
(same claim_id order, just the label column filled in by each person)
"""

import csv
from collections import Counter
def load_labels(path):
    labels = {}
    with open(path, newline="", encoding="cp1252") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_column = [k for k in row.keys() if k.lower().startswith("label")][0]
            labels[row["claim_id"]] = row[label_column].strip().upper()
    return labels

navya = load_labels("label_navya.csv")
reshma = load_labels("label_reshma.csv")
vedika = load_labels("label_vedika.csv")

claim_ids = navya.keys()
agreements = []
consensus = {}
disagreements = []

for cid in claim_ids:
    votes = [navya[cid], reshma[cid], vedika[cid]]
    counts = Counter(votes)
    most_common_label, count = counts.most_common(1)[0]

    if count == 3:
        agreements.append("full")
    elif count == 2:
        agreements.append("majority")
    else:
        agreements.append("none")
        disagreements.append((cid, votes))

    consensus[cid] = most_common_label

full = agreements.count("full")
majority = agreements.count("majority")
none = agreements.count("none")
total = len(agreements)

print(f"Full agreement (3/3): {full}/{total} ({100*full/total:.1f}%)")
print(f"Majority agreement (2/3): {majority}/{total} ({100*majority/total:.1f}%)")
print(f"No agreement (3 different labels): {none}/{total} ({100*none/total:.1f}%)")

if disagreements:
    print(f"\n{len(disagreements)} claims need discussion to resolve:")
    for cid, votes in disagreements:
        print(f"  Claim {cid}: Navya={votes[0]}, Reshma={votes[1]}, Vedika={votes[2]}")

# Save consensus labels (majority vote) as the final dataset
with open("severity_labels_final.csv", "w", newline="", encoding="cp1252") as f:
    writer = csv.writer(f)
    writer.writerow(["claim_id", "consensus_label"])
    for cid, label in consensus.items():
        writer.writerow([cid, label])

print("\nSaved severity_labels_final.csv (majority-vote consensus).")
print("For the 'no agreement' claims, discuss as a team and manually override before final use.")