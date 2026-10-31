# claim_extraction.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import spacy
nlp = spacy.load("en_core_web_sm")

def extract_claims(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
if __name__ == "__main__":
    sample = "Ibuprofen works by inhibiting COX enzymes. It reduces inflammation and pain."
    print(extract_claims(sample))