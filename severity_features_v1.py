"""Frozen text-only feature extraction shared by CV and offline inference."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from severity_scorer import clinical_risk_weight,HIGH_RISK_KEYWORDS,MEDIUM_RISK_KEYWORDS
import re
class MiniLMFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,risk=False):self.risk=risk
    def fit(self,X,y=None):return self
    def transform(self,X):
        from sentence_transformers import SentenceTransformer
        global _encoder
        if '_encoder' not in globals():_encoder=SentenceTransformer('all-MiniLM-L6-v2',device='cpu',local_files_only=True)
        a=_encoder.encode(list(X),show_progress_bar=False,normalize_embeddings=True)
        if self.risk:
            b=np.array([[clinical_risk_weight(t),sum(bool(re.search(r'\b'+re.escape(k)+r'\b',t.lower())) for k in HIGH_RISK_KEYWORDS),sum(bool(re.search(r'\b'+re.escape(k)+r'\b',t.lower())) for k in MEDIUM_RISK_KEYWORDS)] for t in X])
            a=np.column_stack([a,b])
        return a
