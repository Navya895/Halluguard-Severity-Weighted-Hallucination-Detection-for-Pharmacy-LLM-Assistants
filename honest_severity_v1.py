from honest_common_v1 import *
import csv,io,joblib,warnings
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold,GridSearchCV
from sklearn.metrics import f1_score,cohen_kappa_score
from severity_features_v1 import MiniLMFeatures
from severity_scorer import compute_severity
TIERS=['LOW','MEDIUM','HIGH','CRITICAL']
def csvread(name):
    raw=(ROOT/name).read_bytes()
    try:s=raw.decode('utf-8-sig')
    except UnicodeDecodeError:s=raw.decode('cp1252')
    return list(csv.DictReader(io.StringIO(s)))
def score(y,p,groups):
    n=len(y);k=int(sum(y==p));w=int(sum(abs(y-p)<=1))
    result=dict(n=n,correct=k,accuracy=k/n,wilson95=wilson(k,n),within_one=w/n,within_one_wilson95=wilson(w,n),macro_f1=f1_score(y,p,labels=range(4),average='macro',zero_division=0),kappa=cohen_kappa_score(y,p,labels=range(4),weights='quadratic'),confusion=confusion_matrix(y,p,labels=range(4)).tolist())
    rng=np.random.default_rng(42);unique=np.unique(groups);bs=[]
    for _ in range(1000):
        idx=np.concatenate([np.where(groups==g)[0] for g in rng.choice(unique,len(unique),replace=True)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');bs.append([f1_score(y[idx],p[idx],labels=range(4),average='macro',zero_division=0),cohen_kappa_score(y[idx],p[idx],labels=range(4),weights='quadratic')])
    result['cluster_bootstrap95_f1_kappa']=np.nanpercentile(bs,[2.5,97.5],axis=0).T.tolist()
    return result
def main():
    import torch
    torch.set_num_threads(2)
    sheet=csvread('severity_labeling_sheet.csv');nav=csvread('label_navya.csv');labels=csvread('severity_labels_final.csv')
    old=read('train_val_pairs.json');sample=old['train']+old['val'];random.Random(42).shuffle(sample);sample=sample[:50]+[{'claim':r['output'],'context':r['prompt']} for r in read('test_set.json')]
    for name,rows in [('sheet',sheet),('navya',nav),('labels',labels)]:assert len(rows)==58 and len({r['claim_id'] for r in rows})==58,name
    maps=[{int(r['claim_id']):r for r in rows} for rows in [sheet,nav,labels]]
    assert all(set(m)==set(range(58)) for m in maps)
    encoding_repairs=[]
    for i in range(58):
        annotated=maps[1][i]['claim_text']
        if norm(annotated)!=norm(maps[0][i]['claim_text']):
            annotated=annotated.encode('cp1252').decode('utf-8');encoding_repairs.append(i)
        assert norm(maps[0][i]['claim_text'])==norm(annotated)==norm(sample[i]['claim']),f'Text mismatch {i}'
        assert norm(maps[0][i]['context'])==norm(sample[i]['context']),f'Context mismatch {i}'
    X=np.array([maps[0][i]['claim_text'] for i in range(58)]);y=np.array([TIERS.index(maps[2][i]['consensus_label']) for i in range(58)]);groups=np.array([norm(maps[0][i]['context']) for i in range(58)])
    save('severity_alignment.json',{'verified':True,'claims':58,'groups':len(set(groups)),'counts':dict(Counter(TIERS[i] for i in y)),'navya_reversible_encoding_repairs_in_memory_only':encoding_repairs,'checks':['ID uniqueness','exact normalized sheet/Navya/regenerated text after reversible encoding repair','regenerated context']})
    # Cache deterministic frozen embeddings once; no labels used in extraction.
    embeddings=MiniLMFeatures().transform(X);risks=MiniLMFeatures(risk=True).transform(X)
    sources={'tfidf':X,'embedding':embeddings,'embedding_risk':risks}
    def estimator(family):
        return Pipeline([('features',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True) if family=='tfidf' else StandardScaler()),('lr',LogisticRegression(max_iter=2000,random_state=42))])
    all_results={};all_preds={};folds=[]
    for seed in [42,43,44]:
        predictions={k:np.zeros(58,dtype=int) for k in [*sources,'nested_selection','majority']}
        for fold,(tr,te) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(X,y,groups)):
            assert not set(groups[tr])&set(groups[te])
            inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed+100).split(X[tr],y[tr],groups[tr]))
            scores={}
            for family,data in sources.items():
                search=GridSearchCV(estimator(family),{'lr__C':[.1,1,10]},scoring='accuracy',cv=inner,n_jobs=1,error_score='raise').fit(data[tr],y[tr])
                predictions[family][te]=search.predict(data[te]);scores[family]=float(search.best_score_)
                folds.append({'seed':seed,'fold':fold,'family':family,'C':search.best_params_['lr__C'],'inner_score':search.best_score_,'train_ids':tr.tolist(),'test_ids':te.tolist(),'inner_splits':[(tr[a].tolist(),tr[b].tolist()) for a,b in inner]})
            winner=max(scores,key=scores.get);predictions['nested_selection'][te]=predictions[winner][te]
            predictions['majority'][te]=np.bincount(y[tr],minlength=4).argmax()
        all_results[str(seed)]={k:score(y,p,groups) for k,p in predictions.items()};all_preds[str(seed)]={k:p.tolist() for k,p in predictions.items()}
        save('severity_cv.json',all_results);save('severity_predictions.json',{'labels':y.tolist(),'groups':groups.tolist(),'predictions':all_preds});save('severity_folds.json',folds)
        print('severity repeat',seed,{k:fmt(v) for k,v in all_results[str(seed)].items()},flush=True)
    # Select final family/C using internal CV on all data; no outer prediction used to choose it.
    inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=142).split(X,y,groups));final={}
    for family,data in sources.items():
        s=GridSearchCV(estimator(family),{'lr__C':[.1,1,10]},scoring='accuracy',cv=inner,n_jobs=1).fit(data,y);final[family]=(s.best_score_,s.best_params_['lr__C'])
    family=max(final,key=lambda k:final[k][0]);C=final[family][1]
    deployment=Pipeline([('features',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True) if family=='tfidf' else MiniLMFeatures(risk=family=='embedding_risk')),('scale','passthrough' if family=='tfidf' else StandardScaler()),('lr',LogisticRegression(C=C,max_iter=2000,random_state=42))]).fit(X,y)
    folder=OUT/'severity_model';folder.mkdir(exist_ok=True);joblib.dump(deployment,folder/'model.joblib')
    keyword=np.array([TIERS.index(compute_severity(1.,x)['tier']) for x in X])
    save('severity_final.json',{'family':family,'C':C,'full_data_inner_selection':final,'keyword_conditional_p1':score(y,keyword,groups),'note':'Final fit is not an evaluation. Nested-selection CV is headline; Wilson intervals per repeat ignore residual group correlation; bootstrap resamples questions.'})
    print('severity saved',family,C,flush=True)
if __name__=='__main__':main()
