# load_data.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from datasets import load_dataset

ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")["train"]

if __name__ == "__main__":
    print(ds[0])