
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_PATH = "pharmacy_test.json"
MODELS = [
    "./roberta-halluguard-final",
    "./roberta-halluguard-evidence-v2",
]

with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

for model_path in MODELS:
    print("\n" + "=" * 60)
    print("MODEL:", model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    correct = 0

    for item in data:
        inputs = tokenizer(
            item["context"],
            item["claim"],
            return_tensors="pt",
            truncation="only_first",
            max_length=512,
        )

        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0]

        predicted = int(torch.argmax(probs).item())
        actual = int(item["label"])
        correct += predicted == actual

        print(
            f"Actual={actual} Predicted={predicted} "
            f"P0={probs[0].item():.3f} "
            f"P1={probs[1].item():.3f} "
            f"Claim={item['claim'][:75]}"
        )

    print(f"Accuracy: {correct}/{len(data)} = {correct / len(data):.1%}")
