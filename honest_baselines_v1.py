from honest_common_v1 import *
import torch
from transformers import RobertaTokenizer,RobertaForSequenceClassification
from datasets import load_dataset

def main():
    torch.set_num_threads(4)
    protected=list(ROOT.glob('label_*.csv'))+[ROOT/n for n in ['train_val_pairs.json','train_val_pairs_v2.json','train_val_pairs_v3.json','test_set.json','severity_labels_final.csv','severity_labeling_sheet.csv','pharmacy_train.json','pharmacy_test.json']]
    save('protected_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.exists()})
    old=read('train_val_pairs.json');ev=read('train_val_pairs_v3.json')
    assert [(r['claim'],r['label']) for r in old['val']]==[(r['claim'],r['label']) for r in ev['val']]
    overlap=set(norm(r['context']) for r in old['train'])&set(norm(r['context']) for r in old['val'])
    report={'hardware':{'torch':torch.__version__,'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0),'vram_gb':torch.cuda.get_device_properties(0).total_memory/2**30},'historical_train_test_question_overlap':len(overlap),'models':{}}
    for name,rows,length,expected in [('roberta-halluguard-final',old['val'],256,89),('roberta-halluguard-evidence',ev['val'],512,85),('roberta-halluguard-evidence-v2',ev['val'],512,77)]:
        tok=RobertaTokenizer.from_pretrained(str(ROOT/name),local_files_only=True)
        model=RobertaForSequenceClassification.from_pretrained(str(ROOT/name),local_files_only=True).cuda()
        p=predict(model,tok,rows,length);m=metrics([r['label'] for r in rows],p)
        m['expected_correct']=expected;m['matches_expected']=m['correct']==expected
        report['models'][name]=m
        save('baselines.json',report);print(name,fmt(m),flush=True)
        del model;torch.cuda.empty_cache()
    (ROOT/'runs'/'results_summary.md').write_text('# HalluGuard honest evaluation\n\nStatus: baseline audit complete; new experiments pending.\n\nHistorical baselines were previously selected using these validation claims. They are not untouched final-test estimates.\n\n'+f'Historical train/test question overlap: **{len(overlap)} questions**. New training must remove their paired claims too.\n\n'+'\n'.join(f'- {k}: {fmt(v)}; expected count matches: {v["matches_expected"]}; confusion {v["confusion"]}.' for k,v in report['models'].items())+'\n\nReproduce: `.\\.venv\\Scripts\\python.exe honest_baselines_v1.py`\n',encoding='utf-8')
    prepare(old)

def prepare(old):
    from datasets import Dataset
    cache=Path.home()/'.cache/huggingface/datasets/UTAustin-AIHealth___med_hallu/pqa_labeled/0.0.0'
    paths=list(cache.glob('*/med_hallu-train.arrow'))
    assert len(paths)==1,paths
    ds=list(Dataset.from_file(str(paths[0])))
    save('dataset_cache.json',{'path':str(paths[0]),'sha256':hashlib.sha256(paths[0].read_bytes()).hexdigest(),'rows':len(ds)})
    test=old['val'];ph=read('pharmacy_test.json')
    def question(r):return r['context'].split('\nEvidence:')[0].removeprefix('Question:').strip()
    forbidden_q={norm(question(r)) for r in test+ph};forbidden_c={norm(r['claim']) for r in test+ph}
    save('pharmacy_exclusions.json',{'questions':sorted(forbidden_q),'claims':sorted(forbidden_c)})
    eligible=[r for r in ds if norm(r['Question']) not in forbidden_q and not any(norm(r[k]) in forbidden_c for k in ['Ground Truth','Hallucinated Answer'])]
    qs=sorted({r['Question'] for r in eligible});random.Random(42).shuffle(qs);dev=set(qs[:round(len(qs)*.2)])
    splits={'train':[r for r in eligible if r['Question'] not in dev],'dev':[r for r in eligible if r['Question'] in dev]}
    save('detection_split.json',splits)
    save('split_audit.json',{'source_rows':len(ds),'eligible_rows':len(eligible),'excluded_rows':len(ds)-len(eligible),'train_questions':len(splits['train']),'dev_questions':len(splits['dev']),'final_claims':len(test),'pharmacy_train_claims':len(read('pharmacy_train.json')),'pharmacy_test_claims':len(ph),'forbidden_question_count':len(forbidden_q)})
    print('splits',len(splits['train']),len(splits['dev']),flush=True)
if __name__=='__main__':
    import sys
    if '--prepare-only' in sys.argv:prepare(read('train_val_pairs.json'))
    else:main()
