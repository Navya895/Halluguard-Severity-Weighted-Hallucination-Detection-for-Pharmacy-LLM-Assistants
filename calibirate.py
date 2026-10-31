# calibrate.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaTokenizer, RobertaForSequenceClassification

MODEL_PATH = "./roberta-halluguard-final"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

with open("train_val_pairs.json") as f:
    val_data = json.load(f)["val"]

# Collect logits on validation set
all_logits, all_labels = [], []
for ex in val_data:
    inputs = tokenizer(ex["context"], ex["claim"], return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    all_logits.append(logits)
    all_labels.append(ex["label"])

all_logits = torch.cat(all_logits, dim=0)
all_labels = torch.tensor(all_labels)

# Learn a single temperature scalar
temperature = nn.Parameter(torch.ones(1) * 1.5)
optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
nll_loss = nn.CrossEntropyLoss()

def eval_loss():
    optimizer.zero_grad()
    scaled_logits = all_logits / temperature
    loss = nll_loss(scaled_logits, all_labels)
    loss.backward()
    return loss

optimizer.step(eval_loss)
print(f"Learned temperature: {temperature.item():.3f}")

with open("temperature.json", "w") as f:
    json.dump({"temperature": temperature.item()}, f)