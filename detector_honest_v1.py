"""Offline inference for the dev-locked detector, including ensemble calibration."""
import os,json,math
from pathlib import Path
from functools import lru_cache
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/honest_v1'
def configuration():
    packaged=RUN/'final_detector/manifest.json'
    if packaged.exists():return json.loads(packaged.read_text(encoding='utf-8'))
    p=RUN/'selection_LOCKED.json'
    if not p.exists():return None
    selection=json.loads(p.read_text(encoding='utf-8'))
    return {'variant':selection['chosen'],**selection['variants'][selection['chosen']]}
@lru_cache(maxsize=1)
def load_models():
    from transformers import RobertaTokenizer,RobertaForSequenceClassification
    cfg=configuration()
    if cfg is None:raise FileNotFoundError('Detector selection has not been completed')
    return [(RobertaTokenizer.from_pretrained(str(RUN/name),local_files_only=True),RobertaForSequenceClassification.from_pretrained(str(RUN/name),local_files_only=True).eval()) for name in cfg['members']]
def detect(claim,context):
    import torch
    cfg=configuration();values=[]
    for tok,model in load_models():
        x=tok(context,claim,truncation=True,max_length=cfg['length'],return_tensors='pt')
        with torch.no_grad():values.append(float(torch.softmax(model(**x).logits.float(),dim=-1)[0,1]))
    p=max(1e-7,min(1-1e-7,sum(values)/len(values)))
    p=1/(1+math.exp(-math.log(p/(1-p))/cfg['temperature']))
    return p,1-p
