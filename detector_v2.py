# detector_v2.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

MODEL_PATH = "./roberta-halluguard-final"

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


def detect_hallucination(claim, reference_context):
    """
    Returns the probability that the claim is NOT hallucinated (label 1),
    using the fully fine-tuned classifier instead of frozen embeddings + linear probe.
    """
    inputs = tokenizer(reference_context, claim, return_tensors="pt",
                        truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    return probs[0][1].item()  # index 1 = "not_hallucinated" (label=1 in training data)


if __name__ == "__main__":
    claim = "Blue Crystal Syndrome is a rare disorder involving uric acid crystal accumulation."
    reference = "Blue Crystal Syndrome is not a recognized medical diagnosis in any standard medical literature."
    score = detect_hallucination(claim, reference)
    print(f"P(not hallucinated) = {score:.3f}")
    print(f"Predicted: {'not_hallucinated' if score > 0.5 else 'hallucinated'}")