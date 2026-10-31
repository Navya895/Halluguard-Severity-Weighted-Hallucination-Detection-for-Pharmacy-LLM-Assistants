"""Offline, grouped detector experiments. Never reads final labels during training."""
from honest_common_v1 import *
import time, argparse, gc
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, set_seed
from scipy.optimize import minimize_scalar

def pairs(rows,variant,negative=False,seed=42):
    out=[];rng=random.Random(seed)
    for r in rows:
        k=r['Knowledge'];k=' '.join(k) if isinstance(k,list) else k
        context=r['Question']+(' '+k if variant=='evidence' else '')
        for field,label in [('Ground Truth',1),('Hallucinated Answer',0)]:
            out.append(dict(context=context,claim=r[field],label=label,question=r['Question']))
        if negative:
            other=rng.choice([s for s in rows if s['Question']!=r['Question']])
            k=other['Knowledge'];k=' '.join(k) if isinstance(k,list) else k
            out.append(dict(context=r['Question']+' '+k,claim=r['Ground Truth'],label=0,question=r['Question']))
    return out

def train_job(name,train,dev,lr,length,seed,epochs=3,initial='roberta-base',pilot=False):
    set_seed(seed);torch.set_num_threads(4)
    folder=OUT/name;folder.mkdir(exist_ok=True)
    tok=RobertaTokenizer.from_pretrained(initial,local_files_only=True)
    model=RobertaForSequenceClassification.from_pretrained(initial,num_labels=2,local_files_only=True).cuda()
    model.gradient_checkpointing_enable()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01,foreach=False)
    scaler=torch.amp.GradScaler('cuda')
    encoded=tok([r['context'] for r in train],[r['claim'] for r in train],truncation=True,max_length=length)
    best=-1;best_loss=float('inf');history=[];begin=time.time();steps=0
    for epoch in range(epochs):
        model.train();order=np.random.default_rng(seed+epoch).permutation(len(train));opt.zero_grad(set_to_none=True)
        for offset in range(0,len(order),2):
            ids=order[offset:offset+2]
            x=tok.pad([{k:v[int(i)] for k,v in encoded.items()} for i in ids],padding=True,return_tensors='pt').to('cuda')
            y=torch.tensor([train[int(i)]['label'] for i in ids],device='cuda')
            with torch.autocast('cuda',dtype=torch.float16):loss=model(**x,labels=y).loss/4
            scaler.scale(loss).backward();steps+=1
            if steps%4==0 or offset+2>=len(order):
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            if pilot and steps>=8:
                seconds=(time.time()-begin)/steps
                save('pilot.json',{'seconds_per_microbatch':seconds,'length':length,'batch_size':2,'vram_gb':torch.cuda.max_memory_allocated()/2**30})
                return seconds
        p=predict(model,tok,dev,length);m=metrics([r['label'] for r in dev],p);m['epoch']=epoch+1;history.append(m)
        print(name,'epoch',epoch+1,fmt(m),'seconds',round(time.time()-begin),flush=True)
        if (m['accuracy'],-m['log_loss'])>(best,-best_loss):
            best=m['accuracy'];best_loss=m['log_loss'];model.save_pretrained(folder);tok.save_pretrained(folder)
            np.save(folder/'dev_p.npy',p)
        else:break
    result=dict(name=name,seed=seed,lr=lr,length=length,train_claims=len(train),dev_claims=len(dev),history=history,best_accuracy=best,best_log_loss=best_loss,seconds=time.time()-begin)
    save(name+'/training.json',result)
    del model,opt,scaler;gc.collect();torch.cuda.empty_cache()
    return result

def main():
    splits=json.loads((OUT/'detection_split.json').read_text(encoding='utf-8'));start=time.time()
    jobs=[];configs=[(2e-5,256),(1e-5,512)]
    pilot=train_job('pilot',pairs(splits['train'][:8],'evidence',True),pairs(splits['dev'][:2],'evidence'),2e-5,512,42,pilot=True)
    gc.collect();torch.cuda.empty_cache()
    # 2 configurations + 2 additional seeds per variant, 3 epochs each; conservative 30% overhead.
    estimate=pilot*((len(splits['train'])*2+len(splits['train'])*3)/2)*4*3*1.3
    save('runtime_estimate.json',{'total_seconds':estimate,'budget_seconds':18000,'per_job_estimate':'microbatch seconds * ceil(train claims / 2) * 3 epochs * 1.3','pilot_seconds':pilot})
    print('Estimated full grid seconds',estimate,flush=True)
    if estimate>17400:
        save('detection_stopped.json',{'reason':'Projected full experiment exceeds five-hour budget','estimated_seconds':estimate});return
    selection={}
    for variant in ['question','evidence']:
        train=pairs(splits['train'],variant,variant=='evidence');dev=pairs(splits['dev'],variant)
        candidates=[]
        for idx,(lr,length) in enumerate(configs):
            name=f'{variant}_cfg{idx}_seed42'
            print('Job estimate seconds',name,pilot*math.ceil(len(train)/2)*3*1.3,flush=True)
            r=train_job(name,train,dev,lr,length,42);jobs.append(r);candidates.append(r);save('detection_jobs.json',jobs)
        winner=max(candidates,key=lambda r:(r['best_accuracy'],-r['best_log_loss']))
        members=[winner['name']]
        for seed in [43,44]:
            name=f'{variant}_chosen_seed{seed}'
            r=train_job(name,train,dev,winner['lr'],winner['length'],seed);jobs.append(r);members.append(name);save('detection_jobs.json',jobs)
        y=[r['label'] for r in dev];allp=[np.load(OUT/n/'dev_p.npy') for n in members]
        ensemble=np.mean(allp,axis=0);base=allp[0]
        use_ensemble=(metrics(y,ensemble)['accuracy'],-metrics(y,ensemble)['log_loss'])>(metrics(y,base)['accuracy'],-metrics(y,base)['log_loss'])
        p=ensemble if use_ensemble else base
        z=np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7)))
        fit=minimize_scalar(lambda t:log_loss(y,1/(1+np.exp(-z/t)),labels=[0,1]),bounds=(.1,10),method='bounded')
        selection[variant]=dict(members=members if use_ensemble else members[:1],all_seed_members=members,length=winner['length'],lr=winner['lr'],temperature=float(fit.x),dev_before=metrics(y,p),dev_after=metrics(y,1/(1+np.exp(-z/fit.x))),seed_dev=[metrics(y,a) for a in allp],ensemble_dev=metrics(y,ensemble))
        # Optional domain adaptation, grouped pharmacy train/dev, no final pharmacy labels.
        ph=read('pharmacy_train.json') if (ROOT/'pharmacy_train.json').exists() else []
        if ph:
            groups=sorted(set(r['context'].split('\nEvidence:')[0] for r in ph));random.Random(42).shuffle(groups)
            vd=set(groups[:max(1,round(len(groups)*.25))]);pt=[r.copy() for r in ph if r['context'].split('\nEvidence:')[0] not in vd];pd=[r.copy() for r in ph if r['context'].split('\nEvidence:')[0] in vd]
            # Test exclusions checked by preparation audit, without consulting test labels here.
            forbidden=json.loads((OUT/'pharmacy_exclusions.json').read_text(encoding='utf-8'))
            assert all(norm(r['claim']) not in forbidden['claims'] and norm(r['context'].split('\nEvidence:')[0].removeprefix('Question:').strip()) not in forbidden['questions'] for r in pt+pd)
            if variant=='question':
                for r in pt+pd:r['context']=r['context'].split('\nEvidence:')[0].removeprefix('Question:').strip()
            name=variant+'_pharmacy_seed42'
            r=train_job(name,pt,pd,1e-5,winner['length'],42,initial=str(OUT/winner['name']));jobs.append(r);save('detection_jobs.json',jobs)
            # Predeclared conservative acceptance: improve pharmacy dev and do not degrade MedHallu dev.
            tok=RobertaTokenizer.from_pretrained(str(OUT/name),local_files_only=True);model=RobertaForSequenceClassification.from_pretrained(str(OUT/name),local_files_only=True).cuda()
            adapted=predict(model,tok,dev,winner['length']);del model;torch.cuda.empty_cache()
            bp=[]
            for member in selection[variant]['members']:
                model=RobertaForSequenceClassification.from_pretrained(str(OUT/member),local_files_only=True).cuda();bp.append(predict(model,tok,pd,winner['length']));del model;torch.cuda.empty_cache()
            pm=metrics([r['label'] for r in pd],np.mean(bp,axis=0));am=metrics(y,adapted)
            accept=r['best_accuracy']>pm['accuracy'] and am['accuracy']>=selection[variant]['dev_before']['accuracy']
            selection[variant]['pharmacy']={'accepted':accept,'baseline_dev':pm,'adapted_med_dev':am,'name':name,'train_groups':len(groups)-len(vd),'dev_groups':len(vd)}
            if accept:
                z=np.log(np.clip(adapted,1e-7,1-1e-7)/(1-np.clip(adapted,1e-7,1-1e-7)))
                fit=minimize_scalar(lambda t:log_loss(y,1/(1+np.exp(-z/t)),labels=[0,1]),bounds=(.1,10),method='bounded')
                selection[variant].update(members=[name],temperature=float(fit.x),dev_before=am,dev_after=metrics(y,1/(1+np.exp(-z/fit.x))))
        save('selection_in_progress.json',selection)
        if time.time()-start>17400:raise RuntimeError('Five-hour budget guard')
    chosen=max(selection,key=lambda v:(selection[v]['dev_before']['accuracy'],-selection[v]['dev_before']['log_loss']))
    save('selection_LOCKED.json',{'chosen':chosen,'variants':selection,'threshold':.5,'elapsed_seconds':time.time()-start})
    print('SELECTION LOCKED',chosen,flush=True)
if __name__=='__main__':main()
