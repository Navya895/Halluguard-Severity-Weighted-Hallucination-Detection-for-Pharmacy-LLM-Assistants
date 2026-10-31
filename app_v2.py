# app_v2.py
"""
HalluGuard — evidence-aware hallucination analysis prototype.

Expected local project files:
  - llm_connector.py        exposing query_biomistral(prompt, ...)
  - claim_extraction.py     exposing extract_claims(text)
  - ./roberta-halluguard-evidence-v2/

Important:
  * Class 0 is treated as unsupported/contradicted.
  * Class 1 is treated as supported.
  * Probabilities and severity scores are experimental, not clinically validated.
  * Supply trusted reference evidence whenever possible.
"""

import os
import re
from typing import List, Tuple

# Keep model loading offline when local files are present.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

from llm_connector import query_biomistral
try:
    from claim_extraction import extract_claims
except (ImportError, OSError):
    # Allow the offline demo when the optional spaCy language model is absent.
    def extract_claims(text):
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
from severity_scorer_v2 import compute_severity as consensus_severity
from detector_honest_v1 import configuration as honest_configuration, detect as honest_detect


# ---------------------------- Configuration ----------------------------

MODEL_PATH = "./roberta-halluguard-evidence-v2"

# Fixed before new evaluation; historical validation-tuned 0.10 is retired.
THRESHOLD = 0.50
MAX_LENGTH = 512
ENTROPY_SAMPLES = 3

st.set_page_config(
    page_title="HalluGuard",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------- Model loading ----------------------------

@st.cache_resource(show_spinner="Loading HalluGuard classifier...")
def load_finetuned_classifier():
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Model folder not found: {MODEL_PATH}. "
            "Run this app from the project directory containing that folder."
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, local_files_only=True
    )
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading sentence-embedding model...")
def load_embedder():
    # If this model is not cached locally and internet is disabled, semantic
    # entropy will show a helpful error; the core detector can still run.
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------- Text utilities ----------------------------

def clean_generated_text(text: str) -> str:
    """Remove visible chat-template tokens and normalize whitespace."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"<\|[^|]*\|>", " ", text)
    text = re.sub(r"\[/?INST\]", " ", text)
    text = re.sub(r"</?s>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_claims(raw_claims, full_text: str) -> List[str]:
    """Normalize the return value from the existing claim-extraction module."""
    if isinstance(raw_claims, str):
        items = re.split(r"(?<=[.!?])\s+", raw_claims)
    elif isinstance(raw_claims, (list, tuple)):
        items = list(raw_claims)
    else:
        items = []

    claims = []
    seen = set()
    for item in items:
        claim = clean_generated_text(str(item)).strip(" \t\n-*•")
        if len(claim) < 8:
            continue
        if claim.lower() in {"i don't know.", "i don't know", "i do not know."}:
            continue
        key = claim.casefold()
        if key not in seen:
            seen.add(key)
            claims.append(claim)

    # A safe fallback: do not silently report zero claims if extraction failed.
    if not claims:
        fallback = clean_generated_text(full_text)
        if fallback:
            claims = [fallback]
    return claims


def get_generation(prompt: str, temperature: float = 0.2) -> str:
    """Call the project's BioMistral connector, supporting either signature."""
    try:
        result = query_biomistral(prompt, temperature=temperature)
    except TypeError:
        result = query_biomistral(prompt)

    # Accommodate connectors that return a dictionary instead of a string.
    if isinstance(result, dict):
        result = result.get("response") or result.get("text") or result.get("answer") or str(result)
    return clean_generated_text(str(result))


# ---------------------------- Detection ----------------------------

def detect_hallucination(claim: str, reference_context: str) -> Tuple[float, float]:
    """
    Return (P(supported), P(unsupported/contradicted)).

    This is an experimental classifier output, not a calibrated clinical risk.
    The pair order and tokenizer input order match the training/evaluation code:
    tokenizer(context, claim, truncation='only_first').
    """
    if honest_configuration() is not None:
        return honest_detect(claim, reference_context)
    tokenizer, model = load_finetuned_classifier()
    inputs = tokenizer(
        reference_context,
        claim,
        return_tensors="pt",
        truncation="only_first",
        max_length=MAX_LENGTH,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[0]

    p_unsupported = float(probabilities[0].item())
    p_supported = float(probabilities[1].item())
    return p_supported, p_unsupported


# ---------------------------- Severity prioritization ----------------------------

HIGH_RISK_TERMS = (
    "overdose", "fatal", "death", "life-threatening", "contraindicated",
    "allergic reaction", "anaphylaxis", "pregnancy", "pregnant",
    "child", "infant", "stop taking", "discontinue", "double the dose",
    "triple the dose", "dosage", "dose", "drug interaction", "bleeding",
    "kidney failure", "liver failure", "suicid", "poison",
)
MEDIUM_RISK_TERMS = (
    "side effect", "adverse effect", "symptom", "infection", "treatment",
    "medication", "medicine", "antibiotic", "blood pressure", "blood sugar",
    "diagnosis", "prescription", "take with food",
)


def compute_severity(claim: str, p_unsupported: float) -> Tuple[str, float, float]:
    """
    Offline consensus-tier prediction with keyword fallback.
    Returns (tier, priority_score, expected_ordinal_risk_weight).
    Not a clinically validated severity model.
    """
    result = consensus_severity(p_unsupported, claim)
    return result['tier'], result['severity_score'], result['risk_weight']


# ---------------------------- Semantic entropy proxy ----------------------------

def semantic_entropy(prompt: str, n: int = ENTROPY_SAMPLES):
    """
    Embedding-similarity instability proxy. This is NOT true semantic entropy
    and must not be interpreted as a probability of hallucination.
    """
    if n < 2:
        return 0.0, []
    samples = [get_generation(prompt, temperature=0.8) for _ in range(n)]
    samples = [s for s in samples if s]
    if len(samples) < 2:
        return None, samples

    embedder = load_embedder()
    embeddings = embedder.encode(samples, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings)
    n_actual = len(samples)
    upper = sim_matrix[torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()]
    if upper.numel() == 0:
        return 0.0, samples
    mean_similarity = float(upper.mean().item())
    proxy = max(0.0, min(1.0, 1.0 - mean_similarity))
    return proxy, samples


# ---------------------------- UI ----------------------------

st.title("🛡️ HalluGuard")
st.header("Hallucination Detection and Severity Analysis")

st.warning(
    "Research prototype: the RoBERTa detector has not been validated for "
    "reliable pharmacy or medical decisions. Predictions are experimental "
    "and must not be used as medical advice."
)
st.caption(
    "Workflow: generated or pasted response → claim extraction → answer "
    "classification → severity prioritization."
)

question = st.text_area(
    "Enter a question for BioMistral",
    height=85,
    placeholder="Example: Does amoxicillin treat the common cold or flu?",
    key="question_input",
)

reference = st.text_area(
    "Reference evidence (recommended)",
    height=110,
    placeholder=(
        "Paste trusted evidence relevant to the question. "
        "If left blank, the question is used as weak context; "
        "the classifier cannot reliably verify factual support without evidence."
    ),
    key="reference_input",
)

pasted_answer = st.text_area(
    "Answer to analyze (optional; skips BioMistral generation)",
    height=110,
    placeholder="Paste an existing answer to run the detector without Ollama.",
    key="pasted_answer_input",
)

run_entropy = st.checkbox(
    "Also compute response-stability proxy (three additional generations)",
    value=False,
)

with st.expander("How the threshold works"):
    st.write(
        f"The current experimental threshold is **{THRESHOLD:.2f}**. "
        "Claims with P(supported) at or above this value are labelled "
        "'MODEL PREDICTION: SUPPORTED'; other claims are flagged for evidence "
        "review. This threshold was fixed before the new evaluation and is not "
        "a clinically validated cutoff."
    )

if st.button("Analyze", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question first.")
        st.stop()

    if not reference.strip():
        st.info(
            "No reference evidence supplied. The question will be used as "
            "context, so these results should be treated as weak and "
            "inconclusive. For a meaningful test, paste trusted evidence."
        )

    with st.spinner("Generating BioMistral response..."):
        try:
            output = clean_generated_text(pasted_answer) if pasted_answer.strip() else get_generation(question.strip())
        except Exception as exc:
            st.error(f"BioMistral generation failed: {exc}")
            st.stop()

    if not output:
        st.error("BioMistral returned an empty response. Please try again.")
        st.stop()

    st.subheader("1. BioMistral Response")
    st.write(output)

    selected = honest_configuration()
    if selected and selected['variant'] == 'question':
        context = question.strip()
        st.info("The dev-selected detector is question-only: it classifies answer patterns and does not verify the supplied evidence.")
    elif reference.strip():
        context = f"{question.strip()} {reference.strip()}"
    else:
        context = question.strip()

    with st.spinner("Extracting claims and evaluating evidence..."):
        try:
            raw_claims = extract_claims(output)
            claims = normalize_claims(raw_claims, output)
        except Exception as exc:
            st.warning(
                f"Claim extraction failed ({exc}). The complete response "
                "will be evaluated as one claim instead."
            )
            claims = [output]

        claim_results = []
        for claim in claims:
            try:
                p_supported, p_unsupported = detect_hallucination(claim, context)
                tier, priority_score, risk_weight = compute_severity(
                    claim, p_unsupported
                )
                if p_supported >= THRESHOLD:
                    verdict = "MODEL PREDICTION: SUPPORTED"
                else:
                    verdict = "FLAGGED FOR EVIDENCE REVIEW"

                claim_results.append({
                    "claim": claim,
                    "p_supported": p_supported,
                    "p_unsupported": p_unsupported,
                    "verdict": verdict,
                    "severity": tier,
                    "priority_score": priority_score,
                    "risk_weight": risk_weight,
                    "error": None,
                })
            except Exception as exc:
                claim_results.append({
                    "claim": claim,
                    "p_supported": None,
                    "p_unsupported": None,
                    "verdict": "SCORING FAILED",
                    "severity": "UNAVAILABLE",
                    "priority_score": None,
                    "risk_weight": None,
                    "error": str(exc),
                })

    st.subheader("2. Claim-Level Evidence Analysis")
    st.caption(
        "Class 0 = unsupported/contradicted; class 1 = supported, according "
        "to the training labels. The selected model uses internal-dev calibration; "
        "scores remain experimental and are not clinical confidence estimates."
    )

    for idx, result in enumerate(claim_results, start=1):
        with st.container(border=True):
            st.markdown(f"### Claim {idx}")
            st.write(result["claim"])
            st.markdown(f"**Experimental prediction:** {result['verdict']}")

            if result["error"]:
                st.error(f"Could not score this claim: {result['error']}")
                continue

            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    "P(class 1 — supported)",
                    f"{result['p_supported']:.4f}",
                )
            with c2:
                st.metric(
                    "P(class 0 — unsupported/contradicted)",
                    f"{result['p_unsupported']:.4f}",
                )

            st.progress(
                max(0.0, min(1.0, result["p_supported"])),
                text="Probability assigned to the supported class",
            )
            st.write(f"**Conditional severity tier:** {result['severity']}")
            st.write(f"**Severity-priority score:** {result['priority_score']:.3f}")
            st.caption(
                "Severity tier estimates the annotated consequence if the claim is wrong. "
                "Priority weights expected ordinal severity by P(class 0); if the saved "
                "model is absent, keyword scoring is used. Cross-validation did not "
                "demonstrate improvement over the majority baseline."
            )

    scored = [r for r in claim_results if r["error"] is None]
    flagged = [
        r for r in scored
        if r["p_supported"] < THRESHOLD
    ]
    high_critical = [
        r for r in scored
        if r["severity"] in {"HIGH", "CRITICAL"}
    ]

    st.subheader("3. Overall Review Status")
    m1, m2, m3 = st.columns(3)
    m1.metric("Claims analyzed", len(claim_results))
    m2.metric("Claims flagged for review", len(flagged))
    m3.metric("Claims with HIGH/CRITICAL conditional severity", len(high_critical))

    st.error(
        "Human review recommended. Experimental flags are not proof that a "
        "claim is false, and a supported prediction is not proof that a claim "
        "is medically correct."
    )

    if run_entropy:
        st.subheader("4. Response Stability — Similarity Proxy")
        with st.spinner("Generating three additional responses..."):
            try:
                entropy, samples = semantic_entropy(question.strip(), n=ENTROPY_SAMPLES)
                if entropy is None:
                    st.warning("Not enough non-empty generations to calculate the proxy.")
                else:
                    st.metric("Embedding-similarity instability proxy", f"{entropy:.3f}")
                    st.caption(
                        "This is 1 minus average pairwise cosine similarity "
                        "between sampled response embeddings. It is not true "
                        "semantic entropy and is not a calibrated probability "
                        "of hallucination."
                    )
                with st.expander("View sampled generations"):
                    for i, sample in enumerate(samples, start=1):
                        st.markdown(f"**Sample {i}**")
                        st.write(sample or "(empty generation)")
            except Exception as exc:
                st.warning(
                    "The response-stability proxy could not be computed. "
                    f"The core claim analysis is still available. Details: {exc}"
                )

st.divider()
st.caption(
    "HalluGuard is an academic research prototype. Always verify medical "
    "claims against trusted sources and consult qualified healthcare "
    "professionals for medical decisions."
)
