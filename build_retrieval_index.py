# build_retrieval_index.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pickle

ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")["train"]
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_knowledge_text(example):
    k = example["Knowledge"]
    return " ".join(k) if isinstance(k, list) else k

passages, questions = [], []
for ex in ds:
    passages.append(get_knowledge_text(ex))
    questions.append(ex["Question"])

print(f"Embedding {len(passages)} passages...")
passage_embeddings = embedder.encode(passages, show_progress_bar=True, convert_to_numpy=True)

with open("retrieval_index.pkl", "wb") as f:
    pickle.dump({"passages": passages, "questions": questions, "embeddings": passage_embeddings}, f)

print(f"Saved retrieval_index.pkl with {len(passages)} indexed passages.")