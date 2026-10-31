"""Evaluate the current severity prompt independently of hallucination detection."""
import csv
import argparse
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path
from datetime import datetime
import requests

from severity_scorer_llm import DEFAULT_MODEL, SEVERITY_PROMPT, STRONG_PROMPT, query_severity, parse_severity

ROOT = Path(__file__).resolve().parent
TIERS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def read_rows(name):
    data = (ROOT / name).read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252")
    return list(csv.DictReader(io.StringIO(text)))


def summarize(rows):
    matrix = {label: {p: 0 for p in TIERS + ["INVALID"]} for label in TIERS}
    for row in rows:
        matrix[row["human_label"]][row["prediction"]] += 1
    per_class = {}
    for label in TIERS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[actual][label] for actual in TIERS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = dict(precision=precision, recall=recall, f1=f1, support=support)
    return {
        "evaluated": len(rows),
        "accuracy": sum(r["human_label"] == r["prediction"] for r in rows) / len(rows) if rows else 0.0,
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(TIERS),
        "invalid_predictions": sum(r["prediction"] == "INVALID" for r in rows),
        "per_class": per_class,
        "confusion_matrix_rows_actual_columns_predicted": matrix,
    }


def save_metrics(results, output, label_counts):
    metrics = summarize(results)
    metrics["majority_class_accuracy"] = max(label_counts.values()) / len(results)
    extracted = []
    for row in results:
        match = re.match(r"^\s*(?:Severity:\s*)?(LOW|MEDIUM|HIGH|CRITICAL)\b", row["raw_response"], re.IGNORECASE)
        extracted.append({**row, "prediction": match.group(1).upper() if match else "INVALID"})
    metrics["leading_label_extraction"] = summarize(extracted)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--variant", choices=["structured", "strong"], default="structured")
    args = parser.parse_args()
    structured = args.variant == "structured"
    prompt = SEVERITY_PROMPT if structured else STRONG_PROMPT
    labels = read_rows("severity_labels_final.csv")
    source = read_rows("severity_labeling_sheet.csv")
    for name, rows in [("labels", labels), ("claims", source)]:
        ids = [row["claim_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in {name}")
    claims = {row["claim_id"]: row for row in source}
    for row in labels:
        if row["consensus_label"] not in TIERS or row["claim_id"] not in claims:
            raise ValueError(f"Invalid label or missing claim: {row}")
    response = requests.get("http://localhost:11434/api/tags", timeout=10)
    response.raise_for_status()
    installed = {model["name"]: model for model in response.json()["models"]}
    if args.model not in installed:
        raise ValueError(f"Model is not installed: {args.model}")
    output = ROOT / (args.output or f"severity_evaluation_{args.variant}_{datetime.now():%Y%m%d_%H%M%S}")
    output.mkdir(exist_ok=False)
    metadata = {
        "model": args.model, "temperature": 0.1, "seed": 42,
        "model_details": installed[args.model],
        "prompt": prompt,
        "structured": structured,
        "num_predict": 128 if structured else 32,
        "transport": "raw Mistral [INST] template" if args.model.startswith("m/biomistral:") else "native Ollama model template",
        "inputs_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ["severity_labels_final.csv", "severity_labeling_sheet.csv"]},
        "note": "Prompt development evaluation on all existing labels; not an independently held-out performance estimate. Claim-only prompt, no detector input. Invalid outputs count as incorrect.",
        "label_counts": dict(Counter(r["consensus_label"] for r in labels)),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in labels:
            claim = claims[row["claim_id"]]
            raw = query_severity(claim["claim_text"], prompt=prompt, num_predict=metadata["num_predict"], model=args.model)
            normalized = raw.strip().upper().strip(".\"' `\n\r\t")
            prediction = normalized if normalized in TIERS else "INVALID"
            if structured:
                try:
                    prediction = parse_severity(raw)
                except ValueError:
                    prediction = "INVALID"
            result = dict(claim_id=row["claim_id"], claim_text=claim["claim_text"],
                          human_label=row["consensus_label"], prediction=prediction, raw_response=raw)
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{len(results)}/{len(labels)}: {row['claim_id']} human={row['consensus_label']} predicted={prediction}", flush=True)
    metrics = save_metrics(results, output, metadata["label_counts"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
