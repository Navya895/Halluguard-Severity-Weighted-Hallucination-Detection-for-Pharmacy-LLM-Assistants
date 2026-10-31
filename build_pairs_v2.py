# build_pairs_v2.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from datasets import load_dataset
import random
import json
random.seed(42)   # add this line right after the imports, before random.shuffle(examples)

ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")["train"]

pharma_keywords = ["drug", "dose", "dosage", "mg", "therapy", "treatment", "receptor",
                    "inhibitor", "administration", "pharmacokinetic", "adverse", "interaction"]

def is_pharma_relevant(example):
    text = (example["Question"] + " " + example["Ground Truth"]).lower()
    return any(kw in text for kw in pharma_keywords)

pharma_subset = ds.filter(is_pharma_relevant)
print(f"Pharma-relevant rows: {len(pharma_subset)} out of {len(ds)}")

source = pharma_subset if len(pharma_subset) >= 150 else ds
if source is ds:
    print("Pharma subset too small — using full MedHallu set instead.")

def get_knowledge_text(example):
    k = example["Knowledge"]
    return " ".join(k) if isinstance(k, list) else k

examples = []
for ex in source:
    knowledge_text = get_knowledge_text(ex)
    combined_context = ex["Question"] + " " + knowledge_text
    examples.append({"claim": ex["Ground Truth"], "context": combined_context, "label": 1})
    examples.append({"claim": ex["Hallucinated Answer"], "context": combined_context, "label": 0})
random.shuffle(examples)
split = int(0.8 * len(examples))
train_subset, val_subset = examples[:split], examples[split:]

print(f"Total pairs: {len(examples)} | Train: {len(train_subset)} | Val: {len(val_subset)}")

with open("train_val_pairs_v2.json", "w") as f:
    json.dump({"train": train_subset, "val": val_subset}, f)

print("Saved train_val_pairs_v2.json")