"""Package the already-selected detector and inventory completed artifacts."""
from honest_common_v1 import *
import shutil
from honest_report_v1 import main as report
def main():
    assert (OUT/'FINAL_EVALUATION_COMPLETE.json').exists()
    lock=json.loads((OUT/'selection_LOCKED.json').read_text(encoding='utf-8'))
    cfg={'variant':lock['chosen'],**lock['variants'][lock['chosen']]}
    package=OUT/'final_detector';package.mkdir(exist_ok=True)
    members=[]
    for i,name in enumerate(cfg['members']):
        target=package/f'member_{i}'
        if not target.exists():shutil.copytree(OUT/name,target)
        members.append(f'final_detector/member_{i}')
    cfg['original_members']=cfg['members'];cfg['members']=members
    save('final_detector/manifest.json',cfg)
    test=json.loads((OUT/'final_test_results.json').read_text(encoding='utf-8'))
    chosen=test[lock['chosen']]['correct']['after']
    cv=json.loads((OUT/'severity_cv.json').read_text(encoding='utf-8'))
    severity=[r['nested_selection']['accuracy'] for r in cv.values()]
    brief=['# Faculty review: what the evidence supports','',f"The dev-selected detector is **{lock['chosen']}**. Its final MedHallu result is **{fmt(chosen)}**.",'',f"Severity's nested model-selection procedure achieved **{np.mean(severity):.2%} mean exact accuracy**, sample SD {np.std(severity,ddof=1):.2%}; individual-repeat Wilson intervals are in results_summary.md. It did not outperform the majority-class baseline.",'','## How do I know the detector is correct?','','I do not claim that every output is correct. I measured agreement with labeled examples, kept whole questions out of new training and development, selected configurations only on development data, and evaluated the locked models once. I also changed or removed evidence to test whether predictions depend on it. A high benchmark score alone does not prove evidence verification.','','## What changed after the audit?','','The historical 89/94 score reproduces, but 82 questions occurred in both historical training and validation. That score is therefore not a clean unseen-question evaluation. New training excluded all 88 MedHallu test questions and test claims, and started from base RoBERTa rather than the contaminated fine-tuned models.','','## What can I say about severity?','','The 58 labels align with their texts. I used nested cross-validation grouped by question and tested three model families. The labels are imbalanced, with only five CRITICAL examples. The learned system did not show reliable improvement over a majority baseline. Within-one-tier accuracy is a different, easier metric and is not a substitute for exact accuracy. Unanimous annotator agreement is not a mathematical accuracy ceiling.','','## Claims I can honestly make','','- The audit and experiments are reproducible, including failed and weaker runs.','- Confidence intervals communicate the small-sample uncertainty.','- The application is a research prototype with known domain-shift and evidence-verification limitations.','','## Claims I should not make','','- The system is 95% reliable for real pharmacy use.','- A supported classification proves medical correctness.','- These datasets constitute independent clinical validation.','']
    (OUT/'faculty_brief.md').write_text('\n'.join(brief),encoding='utf-8')
    pharmacy=test[lock['chosen']].get('pharmacy_test',{}).get('after')
    if pharmacy:
        with (OUT/'faculty_brief.md').open('a',encoding='utf-8') as f:f.write(f"\nSeparate pharmacy evaluation: **{fmt(pharmacy)}**. This is the key domain-shift limitation; the benchmark score is not pharmacy-wide reliability.\n")
    protected=json.loads((OUT/'protected_hashes.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in protected.items())
    report()
    scripts=sorted(list(ROOT.glob('honest_*_v1.py'))+[ROOT/'severity_features_v1.py',ROOT/'severity_scorer_v2.py',ROOT/'detector_honest_v1.py',ROOT/'app_v2.py'])
    files=scripts+[ROOT/'runs/results_summary.md']+[p for p in OUT.rglob('*') if p.is_file() and p.name!='files_manifest.json']
    print('Packaged detector, report and manifest.',flush=True)
    save('files_manifest.json',{'modified_existing':['app_v2.py'],'protected_files_unchanged':True,'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]})
if __name__=='__main__':main()
