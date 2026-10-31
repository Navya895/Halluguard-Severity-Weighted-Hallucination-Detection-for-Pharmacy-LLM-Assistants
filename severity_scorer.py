# severity_scorer.py
"""
Severity Scorer v0 — a transparent, rule-based first version.

Combines two signals:
1. Hallucination probability (from the fine-tuned RoBERTa classifier)
2. Clinical risk weight (keyword-based) — how dangerous this TYPE of
   claim is if it turns out to be wrong

This is intentionally simple and explainable rather than a trained model,
since no labeled severity dataset exists yet. It's a legitimate, presentable
first-pass module — future work would replace the keyword weighting with
a properly trained severity classifier once labeled data is available.
"""

import re

# Keywords indicating HIGH clinical risk if a claim about them is wrong.
# If a hallucinated claim touches dosage, interactions, or contraindications,
# the real-world consequence of that error is much more severe than, say,
# a wrong historical/discovery fact about a drug.
HIGH_RISK_KEYWORDS = [
    "dose", "dosage", "mg", "overdose", "lethal", "fatal", "toxic", "toxicity",
    "contraindicated", "contraindication", "interaction", "interacts",
    "adverse", "side effect", "allergic", "allergy", "anaphylaxis",
    "pregnancy", "pregnant", "breastfeeding", "renal", "hepatic", "kidney",
    "liver failure", "overdosage", "black box"
]

MEDIUM_RISK_KEYWORDS = [
    "mechanism", "receptor", "inhibitor", "metabolism", "half-life",
    "administration", "indication", "treatment", "therapy", "monitoring"
]

# Everything else (history, discovery, naming, general background facts)
# is treated as lower baseline risk.


def clinical_risk_weight(claim_text):
    """
    Returns a risk weight in [0.3, 1.0] based on keyword presence.
    High-risk keywords dominate; medium-risk keywords give a moderate bump;
    otherwise a low baseline weight is used.
    """
    text = claim_text.lower()

    if any(re.search(r'\b' + re.escape(kw) + r'\b', text) for kw in HIGH_RISK_KEYWORDS):
        return 1.0
    if any(re.search(r'\b' + re.escape(kw) + r'\b', text) for kw in MEDIUM_RISK_KEYWORDS):
        return 0.65
    return 0.3


def compute_severity(hallucination_probability, claim_text):
    """
    Combines hallucination probability with clinical risk weight into
    a single severity score in [0, 1], and maps it to a tier.

    severity_score = hallucination_probability * risk_weight

    This means: a claim can only be HIGH severity if it is BOTH likely
    hallucinated AND touches a high-risk topic. A confidently-fabricated
    but low-risk claim (e.g., a wrong historical fact) stays lower severity
    than a moderately-uncertain but high-risk claim (e.g., a shaky dosage figure).
    """
    risk_weight = clinical_risk_weight(claim_text)
    severity_score = hallucination_probability * risk_weight

    if severity_score >= 0.7:
        tier = "CRITICAL"
    elif severity_score >= 0.45:
        tier = "HIGH"
    elif severity_score >= 0.2:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    return {
        "severity_score": round(severity_score, 3),
        "tier": tier,
        "risk_weight": risk_weight,
        "hallucination_probability": round(hallucination_probability, 3)
    }


if __name__ == "__main__":
    test_cases = [
        ("The maximum daily dose of acetaminophen is 8 grams for adults.", 0.85),
        ("Aspirin was first synthesized in 1897.", 0.85),
        ("Warfarin has no interaction with vitamin K rich foods.", 0.6),
        ("Ibuprofen was discovered by a British pharmaceutical company.", 0.3),
    ]

    for claim, prob in test_cases:
        result = compute_severity(prob, claim)
        print(f"\nClaim: {claim}")
        print(f"  P(hallucinated)={result['hallucination_probability']} | "
              f"Risk weight={result['risk_weight']} | "
              f"Severity={result['severity_score']} | Tier={result['tier']}")