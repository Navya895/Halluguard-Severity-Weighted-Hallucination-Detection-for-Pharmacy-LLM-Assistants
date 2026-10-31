# check_lengths.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
with open("train_val_pairs_v2.json") as f:
    data = json.load(f)

lengths = [len(tokenizer.encode(ex["context"])) for ex in data["train"][:100]]
print(f"Avg context token length: {sum(lengths)/len(lengths):.1f}")
print(f"Max: {max(lengths)}, Min: {min(lengths)}")
print(f"How many exceed 256: {sum(1 for l in lengths if l > 256)}/{len(lengths)}")