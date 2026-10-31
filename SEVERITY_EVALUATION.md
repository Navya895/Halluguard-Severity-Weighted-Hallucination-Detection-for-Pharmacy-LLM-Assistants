# Severity prompt evaluation

All 58 assigned consensus labels were preserved. Detection probabilities were not used.
These are development-set results, not estimates from an independent held-out test set.

| Prompt | Correct | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Original, extracting the leading tier | 14/58 | 24.1% | 0.099 |
| Stronger assumption, one-word output | 4/58 | 6.9% | 0.033 |
| Structured tier and consequence | 18/58 | 31.0% | 0.275 |

The structured prompt is the current default because it improved both accuracy and macro-F1.
Its synthetic controls returned the intended tier for all four explicitly stated consequences.
Those controls test instruction following, not clinical validity.

## Remaining limitations

The selected prompt returned LOW 40 times, MEDIUM 6 times, CRITICAL 12 times, and HIGH zero times.
Recall: LOW 13/14, MEDIUM 2/9, HIGH 0/30, CRITICAL 3/5.
Always predicting HIGH would achieve 51.7% accuracy (macro-F1 approximately 0.170).
The selected prompt improves class-balanced performance over that trivial baseline but remains unreliable,
particularly for HIGH. Prompt selection on these same examples introduces development-set bias.
The model receives claim text only, not the question or reference evidence.
Generated consequences are model outputs and have not been clinically verified.

## Implementation

The scorer uses the local BioMistral model with the raw Mistral instruction template,
temperature 0.1 and seed 42. Structured runs allow 128 generated tokens; one-word runs allow 32.
The public `compute_severity_llm` function still returns a tier string.
It accepts an explicit leading label or `Severity: LABEL`, and raises an error for an invalid response
instead of silently assigning MEDIUM. No global Ollama template or shared connector was changed.

Results and exact prompts are recorded in `severity_evaluation`, `severity_evaluation_v2`,
and `severity_evaluation_v3`. Each has predictions, metrics, and input hashes.

Run the selected prompt again (creates a new timestamped results directory):

```powershell
.\.venv\Scripts\python.exe evaluate_severity.py
```

Reproduce the stronger one-word candidate with `--variant strong`.
Run the four synthetic controls with `probe_severity_prompt.py`.
Run implementation checks with `python -m unittest test_severity_scoring -v`.

The next experiment should compare a different instruction-following model on a fixed development
split, then evaluate the chosen approach on untouched claims. Do not train and report performance
on these same 58 examples as if they were independent test data.
