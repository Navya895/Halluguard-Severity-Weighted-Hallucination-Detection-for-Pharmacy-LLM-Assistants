"""Summarize comparable saved severity runs without calling a model."""
import argparse
import json
from pathlib import Path

from evaluate_severity import TIERS, summarize
from severity_scorer_llm import parse_severity


def load_run(directory):
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    if not rows or len({r["claim_id"] for r in rows}) != len(rows):
        raise ValueError("Empty run or duplicate predictions")
    if len(rows) != sum(metadata["label_counts"].values()):
        raise ValueError("Incomplete run")
    for row in rows:
        try:
            row["prediction"] = parse_severity(row["raw_response"])
        except ValueError:
            row["prediction"] = "INVALID"
    return metadata, rows, summarize(rows)


def compare(baseline, candidate):
    bm, br, bs = load_run(baseline)
    cm, cr, cs = load_run(candidate)
    for key in ("inputs_sha256", "prompt", "temperature", "seed", "num_predict"):
        if bm.get(key) != cm.get(key):
            raise ValueError(f"Runs differ in {key}; not a controlled model comparison")
    actual = {r["claim_id"]: r["human_label"] for r in br}
    if actual != {r["claim_id"]: r["human_label"] for r in cr}:
        raise ValueError("Different examples or labels")
    bp = {r["claim_id"]: r["prediction"] for r in br}
    cp = {r["claim_id"]: r["prediction"] for r in cr}
    improved = [cid for cid in actual if bp[cid] != actual[cid] and cp[cid] == actual[cid]]
    regressed = [cid for cid in actual if bp[cid] == actual[cid] and cp[cid] != actual[cid]]
    return {
        "baseline_model": bm["model"], "candidate_model": cm["model"],
        "baseline": bs, "candidate": cs,
        "improved_claim_ids": improved, "regressed_claim_ids": regressed,
        "note": "Same development claims; not held-out clinical validation. Model-native prompt templates differ.",
    }


def render(report):
    bs, cs = report["baseline"], report["candidate"]
    lines = ["# Severity model comparison", "", report["note"], "",
             f"Baseline: `{report['baseline_model']}`. Candidate: `{report['candidate_model']}`.", "",
             "The assigned labels, claim-only prompt, sampling settings, and parser were held fixed.", "",
             "| Metric | BioMistral | Candidate |", "| --- | ---: | ---: |",
             f"| Accuracy | {bs['accuracy']:.1%} | {cs['accuracy']:.1%} |",
             f"| Macro-F1 | {bs['macro_f1']:.3f} | {cs['macro_f1']:.3f} |",
             f"| Invalid outputs | {bs['invalid_predictions']} | {cs['invalid_predictions']} |", "",
             "| Tier | Examples | Baseline recall | Candidate recall |", "| --- | ---: | ---: | ---: |"]
    for tier in TIERS:
        lines.append(f"| {tier} | {bs['per_class'][tier]['support']} | {bs['per_class'][tier]['recall']:.1%} | {cs['per_class'][tier]['recall']:.1%} |")
    lines += ["", f"Corrected {len(report['improved_claim_ids'])} previous mistakes; introduced {len(report['regressed_claim_ids'])} new mistakes.", "",
              "## Candidate confusion matrix", "", "Rows are assigned labels; columns are predictions.", "",
              "| Actual | LOW | MEDIUM | HIGH | CRITICAL | INVALID |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for tier in TIERS:
        counts = cs["confusion_matrix_rows_actual_columns_predicted"][tier]
        lines.append("| " + " | ".join([tier] + [str(counts[p]) for p in TIERS + ["INVALID"]]) + " |")
    lines += ["", "These results measure agreement with the existing labels. Generated harm explanations have not been clinically verified.",
              "An independent evaluation set is needed before claiming generalization.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=Path("severity_evaluation_v3"))
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate)
    (args.candidate / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.candidate / "comparison.md").write_text(render(report), encoding="utf-8")
    print(render(report))
