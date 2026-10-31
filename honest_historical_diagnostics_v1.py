"""Historical-model ablations; run only after new detector selection is locked."""
from honest_common_v1 import *
import torch
from transformers import RobertaTokenizer,RobertaForSequenceClassification
def main():
    assert (OUT/'selection_LOCKED.json').exists() or (OUT/'detection_stopped.json').exists()
    marker=OUT/'HISTORICAL_DIAGNOSTICS_STARTED'
    with marker.open('x') as f:f.write('No further selection from these diagnostics.')
    torch.set_num_threads(4)
    q=read('train_val_pairs.json')['val'];e=read('train_val_pairs_v3.json')['val'];hand=read('test_set.json');results={}
    for name,rows,length in [('roberta-halluguard-final',q,256),('roberta-halluguard-evidence',e,512),('roberta-halluguard-evidence-v2',e,512)]:
        evidence=name!='roberta-halluguard-final';mismatch=[]
        for i,r in enumerate(rows):
            j=next((i+k)%len(q) for k in range(1,len(q)) if q[(i+k)%len(q)]['context']!=q[i]['context'])
            ev=e[j]['context'][len(q[j]['context']):]
            mismatch.append({**r,'context':q[i]['context']+ev if evidence else q[j]['context']})
        none=[{**r,'context':q[i]['context'] if evidence else ''} for i,r in enumerate(rows)]
        diagnostic=[{'context':r['prompt']+(' '+r.get('reference','') if evidence else ''),'claim':r['output'],'label':{'hallucinated':0,'not_hallucinated':1}[r['label']]} for r in hand]
        tok=RobertaTokenizer.from_pretrained(str(ROOT/name),local_files_only=True);model=RobertaForSequenceClassification.from_pretrained(str(ROOT/name),local_files_only=True).cuda()
        results[name]={}
        for key,rr in [('mismatched',mismatch),('no_evidence',none),('handwritten',diagnostic)]:
            p=predict(model,tok,rr,length);results[name][key]={'metrics':metrics([r['label'] for r in rr],p),'predictions':[{**r,'p_supported':float(a)} for r,a in zip(rr,p)]}
        del model;torch.cuda.empty_cache();save('historical_diagnostics.json',results)
if __name__=='__main__':main()
