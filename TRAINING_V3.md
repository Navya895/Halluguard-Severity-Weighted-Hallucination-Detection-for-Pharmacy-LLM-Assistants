# Training the revised evidence-aware detector

## What the inspection found

The original `train_val_pairs_v3.json` has 82 shared contexts between its training split and its 88 distinct validation contexts. A held-out answer can therefore have its evidence passage represented in training. This makes the old split a weak estimate of generalization to new evidence.

The saved `checkpoint-210/trainer_state.json` reports 88.1% accuracy, 0.80 supported-class F1, 94.4% supported precision and 69.4% supported recall on its historical evaluation set. Those saved metrics are not measurements of the revised code. The saved run evaluated 143 examples, whereas the previous script text selected checkpoints on the original validation split; the artifact and script are not an exact reproducible baseline.

## Changes

- Import PyTorch correctly and run training only from `main()`.
- Rebuild a deterministic split from the old train/validation pool, keeping shared normalized contexts **or claims** together. This does not detect paraphrased duplicates or guarantee that related papers are independent.
- Reject conflicting duplicate labels and remove exact normalized duplicate pairs.
- Reduce synthetic mismatches to 0.5 per supported training example by default. Sample only within that split and exclude known supporting claim/context pairs. Mismatches are assumed unsupported, not verified false; review `synthetic_negatives.json` for label noise.
- Use inverse-frequency class weights to offset augmentation imbalance. The default real-data training set has 279 unsupported and 186 supported examples, with weights 0.8333 and 1.25.
- Use learning rate 1e-5, effective batch size 16 on one device, 10% warmup, weight decay 0.01, and up to six epochs with early stopping after two non-improving evaluations.
- Select the best checkpoint using macro F1 on original grouped validation examples. Also report augmented validation separately, so artificial mismatches do not inflate the primary accuracy.
- Save new runs separately, including exact split, settings, synthetic pairs, validation metrics, predictions, checkpoints and final tokenizer/model.

These settings are starting hypotheses, not proven optimal hyperparameters. Class weighting may trade precision for recall. Macro F1 rewards both classes and need not choose the checkpoint with highest raw accuracy.

## Commands

Run from the project root using the existing Python environment:

```powershell
# Inspect the split without importing/loading ML models:
.\.venv\Scripts\python.exe fiinetune_roberta_v3.py --dry-run

# Train a new model (does not replace the app's existing model):
.\.venv\Scripts\python.exe fiinetune_roberta_v3.py

# Data-integrity tests:
.\.venv\Scripts\python.exe -m unittest test_training_v3 -v

# Small synthetic integration check using random tiny RoBERTa weights:
.\.venv\Scripts\python.exe smoke_training_v3.py
```

Training uses locally cached `roberta-base`; it does not download models. Each run writes `training_runs/evidence-v3-<timestamp>/model/` and adjacent diagnostics. Use `--output-dir <new-empty-directory>` for a chosen destination. The app still uses its existing evidence-v2 model until explicitly updated after evaluation.

Do not initialize from existing fine-tuned project checkpoints when measuring the new split: those weights may already have seen its held-out contexts. The smoke test intentionally substitutes tiny random weights and measures only execution correctness, never accuracy.

## How to measure an improvement

1. Keep the new grouped split fixed (`--seed 42`) across experiments. Leave `pharmacy_test.json` untouched during tuning.
2. Compare `--negative-ratio 0 --unweighted-loss` against the default configuration. This isolates the augmentation/balancing package while retaining the same split and optimization settings.
3. If needed, compare learning rates 1e-5 and 2e-5, changing one setting at a time. Inspect validation macro F1, accuracy and both class recalls. Do not choose settings based on test results.
4. Select the configuration on validation, then evaluate its exported model once on held-out pharmacy data. The existing `eval_zero_shot.py` has a hardcoded evidence-v2 path; change that path deliberately when evaluating a new export. Its pharmacy training split is used for threshold selection, not weight training.
5. Report the test sample size (currently ten examples) and the confusion matrix. One example changes accuracy by ten percentage points. A substantially larger, independently verified, source-held-out dataset is needed for a stable estimate.

The previous reported accuracy and the new grouped-validation accuracy use different splits; a lower new score can reflect the removal of leakage. No accuracy improvement should be claimed until a full run and held-out evaluation substantiate it.

Trainer extension and callback references: [Hugging Face Trainer customization](https://github.com/huggingface/transformers/blob/main/docs/source/en/trainer_customize.md), [Hugging Face callbacks](https://github.com/huggingface/transformers/blob/main/docs/source/en/trainer_callbacks.md). API compatibility was also checked against the installed package.
