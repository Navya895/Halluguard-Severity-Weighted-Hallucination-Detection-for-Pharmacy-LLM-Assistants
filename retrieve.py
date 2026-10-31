# retrieve.py
import pickle
import torch
from sentence_transformers import SentenceTransformer, util

_index = None
_embedder = None

def _load():
    global _index, _embedder
    if _index is None:
        with open("retrieval_index.pkl", "rb") as f:
            _index = pickle.load(f)
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _index, _embedder


def retrieve_context(query, top_k=1):
    index, embedder = _load()
    query_emb = embedder.encode(query, convert_to_tensor=True)
    passage_embs = torch.tensor(index["embeddings"])
    sims = util.cos_sim(query_emb, passage_embs)[0]
    top_indices = torch.topk(sims, k=min(top_k, len(index["passages"]))).indices.tolist()
    passages = [index["passages"][i] for i in top_indices]
    scores = [sims[i].item() for i in top_indices]
    return passages, scores


if __name__ == "__main__":
    test_query = "What is the mechanism of action of a tyrosine kinase inhibitor?"
    passages, scores = retrieve_context(test_query, top_k=1)
    print(f"Query: {test_query}")
    print(f"Best match (similarity={scores[0]:.3f}):")
    print(passages[0][:300], "...")