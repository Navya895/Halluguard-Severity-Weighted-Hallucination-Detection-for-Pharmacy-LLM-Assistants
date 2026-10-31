# llm_connector.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import requests

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


def list_available_models():
    """Quick helper to see the exact model name Ollama has registered."""
    resp = requests.get(OLLAMA_TAGS_URL)
    models = resp.json().get("models", [])
    return [m["name"] for m in models]


def query_biomistral(prompt, model="m/biomistral:latest", temperature=0.7):
    resp = requests.post(OLLAMA_GENERATE_URL, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    })

    data = resp.json()

    if "response" not in data:
        # Surface whatever Ollama actually sent back instead of crashing blindly
        raise RuntimeError(
            f"Ollama did not return a 'response' field.\n"
            f"Full reply: {data}\n"
            f"Available models: {list_available_models()}"
        )

    return data["response"]


if __name__ == "__main__":
    print("Available models:", list_available_models())
    print()
    answer = query_biomistral("What are the diagnostic criteria for Blue Crystal Syndrome?")
    print(answer)