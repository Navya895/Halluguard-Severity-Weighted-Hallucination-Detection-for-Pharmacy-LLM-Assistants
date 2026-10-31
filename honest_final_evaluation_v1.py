"""Single locked final pass. Refuses to evaluate twice in the same run folder."""
from honest_common_v1 import *
import torch
from transformers import RobertaTokenizer,RobertaForSequenceClassification
from datasets import Dataset

def main():
    lock=json.loads((OUT/'selection_LOCKED.json').read_text(encoding='utf-8'))
    marker=OUT/'FINAL_EVALUATION_STARTED.json'
    with marker.open('x',encoding='utf-8') as f:json.dump({'selection_sha256':hashlib.sha256((OUT/'selection_LOCKED.json').read_bytes()).hexdigest()},f)
    torch.set_num_threads(4)
    ds=list(Dataset.from_file(json.loads((OUT/'dataset_cache.json').read_text(encoding='utf-8'))['path']))
    lookup={norm(r[k]):r for r in ds for k in ['Ground Truth','Hallucinated Answer']}
    test=read('train_val_pairs.json')['val'];ph=read('pharmacy_test.json');hand=read('test_set.json')
    results={};predictions={}
    for variant,cfg in lock['variants'].items():
        rows=[]
        for ex in test:
            r=lookup[norm(ex['claim'])];k=r['Knowledge'];k=' '.join(k) if isinstance(k,list) else k
            rows.append({**ex,'question':r['Question'],'evidence':k,'context':r['Question']+(' '+k if variant=='evidence' else ''),'category':r['Category of Hallucination'] if ex['label']==0 else 'SUPPORTED (no hallucination category)','difficulty':r['Difficulty Level']})
        shuffled=[]
        for i,r in enumerate(rows):
            other=next(rows[(i+j)%len(rows)] for j in range(1,len(rows)) if rows[(i+j)%len(rows)]['question']!=r['question'])
            shuffled.append({**r,'context':r['question']+' '+other['evidence'] if variant=='evidence' else other['question']})
        # Question-only ablation swaps question context; evidence variant keeps question and removes evidence.
        none=[{**r,'context':r['question'] if variant=='evidence' else ''} for r in rows]
        pharmacy=[{**r,'context':r['context'] if variant=='evidence' else r['context'].split('\nEvidence:')[0].removeprefix('Question:').strip()} for r in ph]
        sets={'correct':rows,'mismatched':shuffled,'no_evidence':none}
        if pharmacy:sets['pharmacy_test']=pharmacy
        # hand-written diagnostic, labels retained exactly as authored, if available.
        diagnostic=[]
        for r in hand:
            label=r.get('label',r.get('expected_label'))
            label={'hallucinated':0,'not_hallucinated':1,'not hallucinated':1,'supported':1}.get(label,label)
            if label in (0,1):diagnostic.append({'context':r['prompt']+(' '+r.get('reference','') if variant=='evidence' else ''),'claim':r['output'],'label':label})
        if diagnostic:sets['handwritten']=diagnostic
        accum={k:[] for k in sets}
        for member in cfg['members']:
            tok=RobertaTokenizer.from_pretrained(str(OUT/member),local_files_only=True);model=RobertaForSequenceClassification.from_pretrained(str(OUT/member),local_files_only=True).cuda()
            for key,rr in sets.items():accum[key].append(predict(model,tok,rr,cfg['length']))
            del model;torch.cuda.empty_cache()
        results[variant]={};predictions[variant]={}
        for key,rr in sets.items():
            raw=np.mean(accum[key],axis=0);z=np.log(np.clip(raw,1e-7,1-1e-7)/(1-np.clip(raw,1e-7,1-1e-7)));p=1/(1+np.exp(-z/cfg['temperature']))
            results[variant][key]={'before':metrics([r['label'] for r in rr],raw),'after':metrics([r['label'] for r in rr],p)}
            predictions[variant][key]=[{**r,'p_supported_raw':float(a),'p_supported':float(b),'prediction':int(b>=.5)} for r,a,b in zip(rr,raw,p)]
        correct=predictions[variant]['correct'];results[variant]['subgroups']={}
        results[variant]['sensitivity']={}
        for condition in ['mismatched','no_evidence']:
            altered=predictions[variant][condition]
            flips=sum(a['prediction']!=b['prediction'] for a,b in zip(correct,altered))
            results[variant]['sensitivity'][condition]={'flips':flips,'n':len(correct),'flip_rate':flips/len(correct),'flip_wilson95':wilson(flips,len(correct)),'mean_absolute_probability_change':float(np.mean([abs(a['p_supported']-b['p_supported']) for a,b in zip(correct,altered)]))}
        for field in ['category','difficulty']:
            results[variant]['subgroups'][field]={}
            for value in sorted(set(str(r[field]) for r in correct)):
                subset=[r for r in correct if str(r[field])==value];results[variant]['subgroups'][field][value]=metrics([r['label'] for r in subset],[r['p_supported'] for r in subset])
        errors=[r for r in correct if r['label']!=r['prediction']][:10]
        for r in errors:
            i=next(i for i,x in enumerate(correct) if x is r)
            unchanged=all(predictions[variant][condition][i]['prediction']==r['prediction'] for condition in ['mismatched','no_evidence'])
            r['diagnosis']=('False support of a dataset-authored hallucination. ' if r['label']==0 else 'False alarm on a dataset-supported answer. ')+('The erroneous class persists with altered/removed context, consistent with context-insensitive classification.' if unchanged else 'The class changes under a context ablation, showing sensitivity but not reliable verification.')
        results[variant]['first_10_errors']=errors
        save('final_test_results.json',results);save('final_test_predictions.json',predictions)
        print(variant,fmt(results[variant]['correct']['after']),flush=True)
    save('FINAL_EVALUATION_COMPLETE.json',{'chosen':lock['chosen'],'note':'No model/rule changes based on this pass.'})
if __name__=='__main__':main()
