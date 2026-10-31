"""Offline consensus-tier model; keyword fallback if deployment artifact is absent.

The predicted tier is conditional severity of the claim if wrong, as annotated.
The priority score multiplies expected ordinal severity by detector probability.
Semantic entropy is accepted for API compatibility, but is not a trained feature.
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
from pathlib import Path
from functools import lru_cache
from severity_scorer import compute_severity as keyword_severity
MODEL_PATH=Path(__file__).resolve().parent/'runs/honest_v1/severity_model/model.joblib'
TIERS=('LOW','MEDIUM','HIGH','CRITICAL')
@lru_cache(maxsize=1)
def _load_model(path):
    import joblib
    return joblib.load(path)
def compute_severity(hallucination_probability,claim_text,semantic_entropy_score=0.0):
    import math
    probability=float(hallucination_probability)
    if not math.isfinite(probability) or not 0<=probability<=1:
        raise ValueError('hallucination_probability must be finite and between 0 and 1')
    if not MODEL_PATH.is_file():return keyword_severity(probability,claim_text)
    model=_load_model(str(MODEL_PATH));p=model.predict_proba([claim_text])[0]
    classes=model.classes_;tier=int(classes[p.argmax()]);expected=float(sum(float(c)*float(v) for c,v in zip(classes,p))/3)
    return {'severity_score':round(probability*expected,3),'tier':TIERS[tier],'risk_weight':round(expected,3),'hallucination_probability':round(probability,3)}
