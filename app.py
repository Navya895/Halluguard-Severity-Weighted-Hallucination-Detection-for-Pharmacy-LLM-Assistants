# app.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import streamlit as st
import torch
import joblib
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer, util

from llm_connector import query_biomistral
from claim_extraction import extract_claims

# ---------- Load models once, cache across reruns ----------

@st.cache_resource
def load_roberta():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    encoder = RobertaModel.from_pretrained("roberta-base")
    encoder.eval()
    return tokenizer, encoder

@st.cache_resource
def load_classifier():
    return joblib.load("trained_classifier.pkl")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

tokenizer, encoder = load_roberta()
clf = load_classifier()
embedder = load_embedder()


def embed(claim, context):
    inputs = tokenizer(context, claim, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = encoder(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze().numpy()


def detect_hallucination(claim, reference_context):
    emb = embed(claim, reference_context).reshape(1, -1)
    return clf.predict_proba(emb)[0][1]  # prob of "not_hallucinated"


def semantic_entropy(prompt, n=3):
    samples = [query_biomistral(prompt, temperature=0.8) for _ in range(n)]
    embeddings = embedder.encode(samples, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings)
    avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))
    return 1 - avg_sim.item(), samples


# ---------- UI ----------

st.set_page_config(page_title="HalluGuard", page_icon="🩺", layout="wide")
st.title("🩺 HalluGuard — Live Hallucination Detection")
st.caption("Ask BioMistral a pharmacy/medical question. HalluGuard extracts claims, "
           "runs them through a trained classifier, and checks response stability via semantic entropy.")

prompt = st.text_area("Enter your question for BioMistral:", height=80,
                       placeholder="e.g. What is the mechanism of action of ibuprofen?")

reference = st.text_area(
    "Optional: paste a reference/ground-truth fact to check the answer against "
    "(if left blank, the question itself is used as weaker context):",
    height=80,
    placeholder="e.g. Ibuprofen is an NSAID that inhibits COX-1 and COX-2 enzymes..."
)

run_entropy = st.checkbox("Also compute semantic entropy (samples BioMistral 3x — slower)", value=True)

if st.button("Generate & Analyze", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Querying BioMistral..."):
            output = query_biomistral(prompt)

        st.subheader("BioMistral's Response")
        st.write(output)

        context = reference.strip() if reference.strip() else prompt
        if not reference.strip():
            st.info("No reference provided — using the question as context. "
                     "Classifier confidence is weaker without verified ground truth.")

        with st.spinner("Extracting claims and scoring..."):
            claims = extract_claims(output)
            claim_scores = [(c, detect_hallucination(c, context)) for c in claims]

        st.subheader("Per-Claim Analysis")
        overall_consistency = sum(s for _, s in claim_scores) / len(claim_scores) if claim_scores else 0.5
        for claim, score in claim_scores:
            verdict = "✅ Consistent" if score > 0.5 else "⚠️ Likely Hallucinated"
            st.markdown(f"**{verdict}** (P(not hallucinated) = {score:.3f})")
            st.write(f"> {claim}")

        st.subheader("Overall Verdict")
        overall_label = "Not Hallucinated" if overall_consistency > 0.5 else "Hallucinated"
        st.metric("Hallucination Probability", f"{1 - overall_consistency:.3f}", label_visibility="visible")
        st.write(f"**Prediction: {overall_label}**")

        if run_entropy:
            with st.spinner("Sampling BioMistral 3x for semantic entropy (this takes a bit)..."):
                entropy, samples = semantic_entropy(prompt, n=3)
            st.subheader("Semantic Entropy (Response Stability)")
            st.write(f"Entropy score: **{entropy:.3f}** "
                     f"({'low — model is consistent across samples' if entropy < 0.4 else 'high — model is unstable/uncertain across samples'})")
            with st.expander("View the 3 sampled generations"):
                for i, s in enumerate(samples, 1):
                    st.markdown(f"**Sample {i}:** {s}")