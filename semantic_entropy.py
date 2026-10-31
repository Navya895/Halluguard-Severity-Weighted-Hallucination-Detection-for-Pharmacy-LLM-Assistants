# semantic_entropy.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from llm_connector import query_biomistral
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_entropy(prompt, n=3):
    samples = [query_biomistral(prompt, temperature=0.8) for _ in range(n)]
    embeddings = embedder.encode(samples, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings)
    avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))
    return 1 - avg_sim.item()

if __name__ == "__main__":
    prompt = "What are the diagnostic criteria for Blue Crystal Syndrome?"
    print("Semantic entropy:", semantic_entropy(prompt))