
"""
HalluGuard: Pharmacy Dataset Builder

Input:
    pharmacy_dataset.csv
    pharmacy_dataset_template.csv

Required CSV columns:
    id, drug, question, evidence, true_claim,
    false_claim, source_url, annotator

Output:
    pharmacy_train.json
    pharmacy_test.json

Labels:
    1 = supported by evidence
    0 = contradicted by evidence

IMPORTANT:
This script does not invent medical records.
Only manually verified, complete records are included.
The test split holds out entire drugs when possible.
"""

import csv
import json
import random
from pathlib import Path
from collections import defaultdict

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CSV_CANDIDATES = [
    Path("pharmacy_dataset.csv"),
    Path("pharmacy_dataset_annotated.csv"),
]

TEST_FRACTION = 0.25
SEED = 42

REQUIRED_COLUMNS = [
    "drug",
    "question",
    "evidence",
    "true_claim",
    "false_claim",
]

OUTPUT_TRAIN = Path("pharmacy_train.json")
OUTPUT_TEST = Path("pharmacy_test.json")


# --------------------------------------------------
# READ CSV
# --------------------------------------------------

def read_csv(path):
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with open(
                path, "r", newline="", encoding=encoding
            ) as file:
                reader = csv.DictReader(file)

                if not reader.fieldnames:
                    return [], []

                # Strip whitespace from column headers.
                reader.fieldnames = [
                    name.strip() if name else name
                    for name in reader.fieldnames
                ]

                rows = list(reader)
                return reader.fieldnames, rows

        except UnicodeDecodeError:
            continue

    raise ValueError(f"Cannot decode CSV: {path}")


# --------------------------------------------------
# FIND A POPULATED DATASET
# --------------------------------------------------

def find_dataset():
    found_files = []

    for path in CSV_CANDIDATES:
        if not path.exists():
            continue

        if path.stat().st_size == 0:
            print(f"Skipping empty file: {path}")
            continue

        headers, rows = read_csv(path)

        if not headers:
            print(f"Skipping CSV without headers: {path}")
            continue

        missing = set(REQUIRED_COLUMNS) - set(headers)

        if missing:
            print(
                f"Skipping {path}: missing columns "
                f"{sorted(missing)}"
            )
            continue

        # Do not select a file just because it has example rows.
        real_rows = [
            row for row in rows
            if not (
                (row.get("id") or "")
                .strip()
                .upper()
                .startswith("EXAMPLE")
            )
        ]

        complete_rows = [
            row for row in real_rows
            if all(
                (row.get(column) or "").strip()
                for column in REQUIRED_COLUMNS
            )
        ]

        found_files.append(
            (path, headers, rows, complete_rows)
        )

    # Prefer the file with the most complete real records.
    if not found_files:
        raise FileNotFoundError(
            "\nNo usable pharmacy CSV was found.\n"
            "Create real, verified records in "
            "pharmacy_dataset_template.csv.\n"
            "Keep the required column names unchanged."
        )

    found_files.sort(
        key=lambda item: len(item[3]),
        reverse=True,
    )

    path, headers, rows, clean = found_files[0]

    print(f"\nSelected input: {path.resolve()}")
    print(f"CSV rows: {len(rows)}")
    print(f"Complete non-example rows: {len(clean)}")

    if not clean:
        raise ValueError(
            "\nNo usable pharmacy records were found.\n"
            "Your current CSV appears to contain only examples "
            "or incomplete rows.\n"
            "Add verified records with drug, question, evidence, "
            "true_claim and false_claim before generating data.\n"
            "No empty training files were written."
        )

    return clean


# --------------------------------------------------
# VALIDATE RECORDS
# --------------------------------------------------

def validate_rows(rows):
    clean = []
    seen_ids = set()
    problems = []

    for line_number, row in enumerate(rows, start=2):
        record_id = (row.get("id") or "").strip()
        drug = (row.get("drug") or "").strip().lower()

        missing = [
            column for column in REQUIRED_COLUMNS
            if not (row.get(column) or "").strip()
        ]

        if missing:
            problems.append(
                f"CSV line {line_number}: missing {missing}"
            )
            continue

        if not record_id:
            problems.append(
                f"CSV line {line_number}: missing ID"
            )
            continue

        if record_id in seen_ids:
            problems.append(
                f"CSV line {line_number}: duplicate ID {record_id}"
            )
            continue

        if not drug:
            problems.append(
                f"CSV line {line_number}: missing drug"
            )
            continue

        if (
            row["true_claim"].strip()
            == row["false_claim"].strip()
        ):
            problems.append(
                f"CSV line {line_number}: true and false claims "
                "are identical"
            )
            continue

        seen_ids.add(record_id)
        clean.append(row)

    if problems:
        print("\nRecord validation warnings:")
        for problem in problems:
            print("-", problem)

    if not clean:
        raise ValueError(
            "No records passed validation. "
            "Correct the CSV before continuing."
        )

    return clean


# --------------------------------------------------
# CONVERT EACH ROW INTO TWO CLAIMS
# --------------------------------------------------

def make_pairs(row):
    question = row["question"].strip()
    evidence = row["evidence"].strip()

    context = f"Question: {question}\nEvidence: {evidence}"

    return [
        {
            "claim": row["true_claim"].strip(),
            "context": context,
            "label": 1,
        },
        {
            "claim": row["false_claim"].strip(),
            "context": context,
            "label": 0,
        },
    ]


# --------------------------------------------------
# SPLIT BY DRUG
# --------------------------------------------------

def split_by_drug(rows):
    by_drug = defaultdict(list)

    for row in rows:
        drug = row["drug"].strip().lower()
        by_drug[drug].append(row)

    drugs = sorted(by_drug)

    if len(drugs) < 2:
        raise ValueError(
            f"Only {len(drugs)} distinct drug(s) found. "
            "At least two are needed for a drug-held-out split. "
            "Add verified records covering more drugs."
        )

    random.Random(SEED).shuffle(drugs)

    n_test = max(
        1,
        round(len(drugs) * TEST_FRACTION),
    )

    # Ensure at least one drug remains in training.
    n_test = min(n_test, len(drugs) - 1)

    test_drugs = set(drugs[:n_test])

    train = []
    test = []

    for drug, drug_rows in by_drug.items():
        destination = test if drug in test_drugs else train

        for row in drug_rows:
            destination.extend(make_pairs(row))

    if not train or not test:
        raise ValueError(
            "Train/test split is empty. "
            "Check the number of distinct drugs."
        )

    return train, test, drugs, test_drugs


# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main():
    print("=" * 55)
    print("HALLUGUARD PHARMACY DATASET BUILDER")
    print("=" * 55)

    rows = find_dataset()
    rows = validate_rows(rows)

    train, test, drugs, test_drugs = split_by_drug(rows)

    # Check the output format expected by eval_zero_shot.py.
    for name, dataset in [("train", train), ("test", test)]:
        if not dataset:
            raise ValueError(f"{name} dataset is empty.")

        labels = {item["label"] for item in dataset}

        if labels != {0, 1}:
            raise ValueError(
                f"{name} must contain both labels 0 and 1."
            )

    save_json(OUTPUT_TRAIN, train)
    save_json(OUTPUT_TEST, test)

    print("\nDataset generation successful.")
    print(f"Verified source rows: {len(rows)}")
    print(f"Distinct drugs: {len(drugs)}")
    print(f"Held-out test drugs: {sorted(test_drugs)}")
    print(f"Training examples: {len(train)}")
    print(f"Testing examples: {len(test)}")

    print(
        "Training labels:",
        {label: sum(x["label"] == label for x in train)
         for label in (0, 1)},
    )

    print(
        "Testing labels:",
        {label: sum(x["label"] == label for x in test)
         for label in (0, 1)},
    )

    print(f"\nSaved: {OUTPUT_TRAIN.resolve()}")
    print(f"Saved: {OUTPUT_TEST.resolve()}")


if __name__ == "__main__":
    main()
