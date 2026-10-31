"""Independent synthetic controls for the severity instruction, not clinical validation."""
import json
import argparse
from pathlib import Path
from datetime import datetime
from severity_scorer_llm import DEFAULT_MODEL, query_severity, parse_severity, SEVERITY_PROMPT as PROMPT

CONTROLS = [
    ("LOW", "An incorrect date in a medicine history exhibit changes no diagnosis, treatment, or patient behavior."),
    ("MEDIUM", "An incorrect classroom explanation of a biological mechanism misleads understanding but changes no treatment and causes no direct physical harm."),
    ("HIGH", "An incorrect treatment instruction causes significant temporary harm requiring hospitalization, followed by full recovery; no death or permanent injury occurs."),
    ("CRITICAL", "An incorrect emergency treatment instruction directly causes a patient's death when followed."),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = []
    for expected, claim in CONTROLS:
        raw = query_severity(claim, model=args.model)
        try:
            predicted = parse_severity(raw)
        except ValueError:
            predicted = "INVALID"
        row = {"expected": expected, "claim": claim, "response": raw, "predicted": predicted}
        rows.append(row)
        print(json.dumps(row), flush=True)
    output = Path(__file__).parent / (args.output or f"severity_controls_{datetime.now():%Y%m%d_%H%M%S}.json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"model": args.model, "prompt": PROMPT, "results": rows}, stream, indent=2)
