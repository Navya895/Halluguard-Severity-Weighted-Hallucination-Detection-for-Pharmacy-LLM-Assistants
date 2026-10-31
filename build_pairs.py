# build_pairs.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from datasets import load_dataset
import random

ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")["train"]

pharma_keywords = ["drug", "dose", "dosage", "mg", "therapy", "treatment", "receptor",
                    "inhibitor", "administration", "pharmacokinetic", "adverse", "interaction"]

def is_pharma_relevant(example):
    text = (example["Question"] + " " + example["Ground Truth"]).lower()
    return any(kw in text for kw in pharma_keywords)

pharma_subset = ds.filter(is_pharma_relevant)
print(f"Pharma-relevant rows: {len(pharma_subset)} out of {len(ds)}")

# Fall back to full dataset if the filtered subset is too small
source = pharma_subset if len(pharma_subset) >= 150 else ds
if source is ds:
    print("Pharma subset too small — using full MedHallu set instead.")

examples = []
for ex in source:
    examples.append({"claim": ex["Ground Truth"], "context": ex["Question"], "label": 1})
    examples.append({"claim": ex["Hallucinated Answer"], "context": ex["Question"], "label": 0})

random.shuffle(examples)
split = int(0.8 * len(examples))
train_subset, val_subset = examples[:split], examples[split:]

print(f"Total pairs: {len(examples)} | Train: {len(train_subset)} | Val: {len(val_subset)}")

if __name__ == "__main__":
    import json
    with open("train_val_pairs.json", "w") as f:
        json.dump({"train": train_subset, "val": val_subset}, f)
    print("Saved train_val_pairs.json")