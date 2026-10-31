import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json, math, hashlib, random
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, log_loss
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs'/'honest_v1'
OUT.mkdir(parents=True,exist_ok=True)
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def read(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))
def norm(x): return ' '.join(str(x).casefold().split())
def wilson(k,n):
    if not n:return [None,None]
    z=1.95996398454;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-w,c+w]
def metrics(y,p):
    y=np.array(y);p=np.asarray(p);pred=(p>=.5).astype(int);k=int(sum(y==pred))
    ece=0.
    for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        m=(p>=a)&((p<b) if b<1 else (p<=b))
        if m.any(): ece+=m.mean()*abs(y[m].mean()-p[m].mean())
    return dict(n=len(y),correct=k,accuracy=k/len(y),wilson95=wilson(k,len(y)),auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,ece=float(ece),log_loss=float(log_loss(y,p,labels=[0,1])),confusion=confusion_matrix(y,pred,labels=[0,1]).tolist())
def summary_append(text):
    with (ROOT/'runs'/'results_summary.md').open('a',encoding='utf-8') as f:f.write(text+'\n')
def fmt(m):return f"{m['correct']}/{m['n']} = {m['accuracy']:.2%} (Wilson 95% {m['wilson95'][0]:.2%}–{m['wilson95'][1]:.2%})"
def predict(model,tok,rows,length=256):
    import torch
    model.eval();ps=[]
    for start in range(0,len(rows),4):
        b=rows[start:start+4]
        x=tok([r['context'] for r in b],[r['claim'] for r in b],padding=True,truncation=True,max_length=length,return_tensors='pt').to(next(model.parameters()).device)
        with torch.no_grad():ps.extend(torch.softmax(model(**x).logits.float(),-1)[:,1].cpu().tolist())
    return np.array(ps)
