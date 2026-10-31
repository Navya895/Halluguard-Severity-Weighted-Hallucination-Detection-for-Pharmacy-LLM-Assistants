# finetune_roberta_v2.py
"""
Full fine-tuning of RoBERTa on EVIDENCE-GROUNDED context (Question + Knowledge),
max_length=512. Saved to a NEW folder so the existing 94.7% model is untouched.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import numpy as np
from datasets import Dataset
from transformers import (RobertaTokenizer, RobertaForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

MAX_LEN = 512
OUT_DIR = "./roberta-halluguard-evidence"

with open("train_val_pairs_v3.json") as f:
    data = json.load(f)

train_ds = Dataset.from_list(data["train"])
val_ds = Dataset.from_list(data["val"])
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def preprocess(batch):
    # only_first: truncate the evidence if too long, never cut the claim itself
    return tokenizer(batch["context"], batch["claim"],
                     truncation="only_first", max_length=MAX_LEN)


train_ds = train_ds.map(preprocess, batched=True).rename_column("label", "labels")
val_ds = val_ds.map(preprocess, batched=True).rename_column("label", "labels")
keep = ["input_ids", "attention_mask", "labels"]
train_ds.set_format("torch", columns=keep)
val_ds.set_format("torch", columns=keep)

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": accuracy_score(labels, preds), "f1": f1,
            "precision": p, "recall": r}


args = TrainingArguments(
    output_dir="./roberta-halluguard-evidence-checkpoints",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,      # effective batch size 8
    num_train_epochs=4,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=20,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

print("\nStarting evidence-grounded fine-tuning (this is slow on CPU — leave it running)...\n")
trainer.train()

print("\n=== Final Evaluation (evidence-grounded model) ===")
print(trainer.evaluate())

pred = trainer.predict(val_ds)
y_pred = np.argmax(pred.predictions, axis=1)
print("\n=== Confusion Matrix ===")
print(confusion_matrix(pred.label_ids, y_pred))

trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"\nSaved to {OUT_DIR}")
print("Compare against the question-only model: accuracy 94.7%, F1 0.948, confusion [[43,2],[3,46]]")