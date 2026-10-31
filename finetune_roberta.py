# finetune_roberta.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import numpy as np
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ---------- Load the same pairs used for the linear-probe baseline ----------
with open("train_val_pairs.json") as f:
    data = json.load(f)

train_ds = Dataset.from_list(data["train"])
val_ds = Dataset.from_list(data["val"])

print(f"Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")

# ---------- Tokenize ----------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def preprocess(batch):
    return tokenizer(batch["context"], batch["claim"], truncation=True,
                      padding="max_length", max_length=256)

train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)

train_ds = train_ds.rename_column("label", "labels")
val_ds = val_ds.rename_column("label", "labels")

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ---------- Model (this time NOT frozen — full fine-tuning) ----------
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

args = TrainingArguments(
    output_dir="./roberta-halluguard-finetuned",
    per_device_train_batch_size=8,      # smaller batch size — safer on CPU/limited RAM
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=20,
    report_to="none",                    # disables wandb/etc. prompts
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

print("\nStarting fine-tuning...\n")
trainer.train()

print("\n=== Final Evaluation ===")
eval_results = trainer.evaluate()
print(eval_results)

# ---------- Confusion matrix on val set ----------
predictions = trainer.predict(val_ds)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred))

# ---------- Save the fine-tuned model ----------
trainer.save_model("./roberta-halluguard-final")
tokenizer.save_pretrained("./roberta-halluguard-final")
print("\nSaved fine-tuned model to ./roberta-halluguard-final")