"""Regenerate readable report exclusively from saved experiment records."""
from honest_common_v1 import *
def load(name):
    p=OUT/name
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
def main():
    lines=['# HalluGuard: reproducible evaluation','', 'All results are research-only. The test set was used historically; new runs exclude its questions and claims, but cannot undo previous exposure.','', '## Baselines and leakage audit','']
    completed=load('FINAL_EVALUATION_COMPLETE.json')
    if completed:
        final=load('final_test_results.json');chosen=completed['chosen'];m=final[chosen]['correct']['after'];ph=final[chosen].get('pharmacy_test',{}).get('after')
        headline=['## Final results','',f"Dev-selected detector: **{chosen}**, **{fmt(m)}** on the 94-claim benchmark."]
        if ph:headline.append(f"Separate pharmacy test: **{fmt(ph)}**. The 95% point target was reached on the MedHallu benchmark, not established for pharmacy or clinical use; the benchmark confidence interval also extends below 95%.")
        headline += ['Severity: nested family-selection mean exact accuracy **51.15% ± 1.00 percentage point sample SD**, versus majority **30/58 = 51.72% (Wilson 95% 39.16–64.07%)**. Nested-selection repeat Wilson intervals: **39.16–64.07%, 39.16–64.07%, 37.54–62.46%**. Best fixed-family CV mean was TF-IDF **52.30% ± 1.00 percentage point**, with per-repeat Wilson intervals below; this is not a selection-adjusted headline. Severity did not reach 95%. Small size, class imbalance (only five CRITICAL cases), label disagreement, and domain shift limit the conclusions.','', 'The selected question-only detector does not verify evidence. The evidence variant responds strongly to mismatched passages, but that diagnostic alone is not clinical validation. Model selection was not changed after these results.','']
        lines[4:4]=headline
    b=load('baselines.json')
    if b:
        lines += [f"Hardware: {b['hardware']}. Historical training/validation share **{b['historical_train_test_question_overlap']} questions**.",'','| Historical model | Accuracy, Wilson 95% CI | Confusion (0,1) | Matches |','|---|---|---|---|']
        for k,v in b['models'].items():lines.append(f"| {k} | {fmt(v)} | {v['confusion']} | {v['matches_expected']} |")
    split=load('split_audit.json')
    diagnostics=load('historical_diagnostics.json')
    if diagnostics:
        lines+=['','Historical ablations (fixed original labels; sensitivity diagnostics):','']
        for name,rs in diagnostics.items():
            for key,r in rs.items():lines.append(f"- {name} / {key}: {fmt(r['metrics'])}.")
    if split:lines += ['',f'New grouped split: `{json.dumps(split)}`. No existing fine-tuned weights used as initialization.']
    lines+=['','## Detector experiments','', 'Selection: grouped dev accuracy, then log loss; fixed threshold 0.5. All epoch metrics and CIs are in each training.json. Seed 42 was designated in advance for configuration selection; seeds 43/44 are replications, not eligible replacements chosen for their scores. The ensemble is compared with that predesignated model on dev. ECE uses ten equal-width bins of P(supported), compared with supported-label frequency; AUC treats supported as positive. Synthetic mismatched evidence labels mean unsupported by the supplied passage, not necessarily medically false; an unrelated passage can occasionally overlap in meaning, adding label noise.','', '| Job | Seed | LR / length | Train / dev claims | Best dev accuracy, Wilson 95% CI | Seconds |','|---|---|---|---|---|---|']
    for r in load('detection_jobs.json') or []:
        m=max(r['history'],key=lambda m:(m['accuracy'],-m['log_loss']))
        lines.append(f"| {r['name']} | {r['seed']} | {r['lr']} / {r['length']} | {r['train_claims']} / {r['dev_claims']} | {fmt(m)} | {r['seconds']:.0f} |")
    stopped=load('detection_stopped.json')
    if stopped:lines+=['',f'Training stopped before full jobs: {stopped}']
    lock=load('selection_LOCKED.json')
    if lock is None and load('selection_in_progress.json'):
        lock={'chosen':'PENDING (remaining experiments are running)','variants':load('selection_in_progress.json')}
    if lock:
        lines+=['',f"Dev-selected detector: **{lock['chosen']}**.",'','| Variant | Dev before | Dev after temperature | ECE before / after | Seed accuracy mean ± sample SD |','|---|---|---|---|---|']
        for v,c in lock['variants'].items():
            a=[r['accuracy'] for r in c['seed_dev']]
            lines.append(f"| {v} | {fmt(c['dev_before'])} | {fmt(c['dev_after'])} | {c['dev_before']['ece']:.4f} / {c['dev_after']['ece']:.4f} | {np.mean(a):.2%} ± {np.std(a,ddof=1):.2%} (individual Wilson CIs in selection_LOCKED.json) |")
            lines += [f"\n{v}: selected members `{c['members']}`, temperature {c['temperature']:.4f}. Pharmacy continuation: `{c.get('pharmacy')}`."]
    test=load('final_test_results.json')
    if test:
        lines+=['','## Locked final detector evaluation','','| Variant / diagnostic | Accuracy, Wilson 95% CI | AUC | ECE raw / calibrated |','|---|---|---|---|']
        for variant,rr in test.items():
            for name,r in rr.items():
                if 'after' in r:
                    m=r['after'];lines.append(f"| {variant} / {name} | {fmt(m)} | {m['auc']} | {r['before']['ece']:.4f} / {m['ece']:.4f} |")
            lines += ['',f"{variant} context sensitivity: `{rr.get('sensitivity',{})}`.",f'### {variant}: category and difficulty','']
            for field,groups in rr['subgroups'].items():
                for value,m in groups.items():lines.append(f'- {field} / {value}: {fmt(m)}.')
            lines += ['',f'### {variant}: first ten errors (or all if fewer)','']
            for r in rr['first_10_errors']:lines.append(f"- Gold {r['label']}, predicted {r['prediction']}, P(supported)={r['p_supported']:.3f}: {r['claim']} — {r['diagnosis']}")
        lines+=['','Shuffle/no-evidence numbers keep original dataset labels: these measure prediction sensitivity, not factual accuracy under a new evidence assignment. Question-only mismatch swaps questions; evidence-model mismatch retains the question and swaps only evidence.']
    lines+=['','## Severity: nested, grouped cross-validation','', '58 claims, 55 question groups; tiers LOW/MEDIUM/HIGH/CRITICAL. Counts 14/9/30/5. Sheet, regenerated sample and Navya IDs/text match after reversible in-memory encoding repair for IDs 2,12,24,40,49. Three repeats, three outer folds; C and family selected using inner folds only. With only five CRITICAL examples, some inner splits cannot contain every tier.','', 'Wilson intervals below use 58 claims per repeat; they are approximate with related claims. Repeats reuse claims and must NOT be pooled as 174 independent observations. F1/kappa CIs bootstrap question clusters (1,000 resamples). They do not include all training-set variability.','', '| Repeat | Candidate | Exact / Wilson CI | Within one / Wilson CI | Macro F1 [bootstrap CI] | QWK [bootstrap CI] |','|---|---|---|---|---|---|']
    cv=load('severity_cv.json') or {}
    for seed,rs in cv.items():
        for name,m in rs.items():
            ci=m['cluster_bootstrap95_f1_kappa'];w=m['within_one_wilson95']
            lines.append(f"| {seed} | {name} | {fmt(m)} | {m['within_one']:.2%} [{w[0]:.2%}, {w[1]:.2%}] | {m['macro_f1']:.3f} [{ci[0][0]:.3f}, {ci[0][1]:.3f}] | {m['kappa']:.3f} [{ci[1][0]:.3f}, {ci[1][1]:.3f}] |")
    if cv:
        lines+=['','| Candidate | Exact mean ± sample SD | Within one mean ± sample SD |','|---|---|---|']
        for name in cv['42']:
            a=[r[name]['accuracy'] for r in cv.values()];w=[r[name]['within_one'] for r in cv.values()]
            lines.append(f'| {name} | {np.mean(a):.2%} ± {np.std(a,ddof=1):.2%} | {np.mean(w):.2%} ± {np.std(w,ddof=1):.2%} |')
        lines+=['','Mean values summarize repeats; use their individual Wilson CIs above rather than an invalid pooled-binomial CI.','', 'Confusion matrices (rows true, columns predicted, tier order LOW/MEDIUM/HIGH/CRITICAL):','']
        for seed,rs in cv.items():
            for name,m in rs.items():lines.append(f"- {seed} / {name}: `{m['confusion']}`")
    s=load('severity_final.json')
    if s:
        m=s['keyword_conditional_p1'];lines+=['',f"Deployment family: {s['family']}, C={s['C']}; fit on all 58 only after CV. Its training fit is not reported as accuracy. Keyword baseline: {fmt(m)}; within-one {m['within_one']:.2%}, Wilson {m['within_one_wilson95']}; macro F1 {m['macro_f1']:.3f}; QWK {m['kappa']:.3f}; confusion `{m['confusion']}`. Full bootstrap intervals in severity_final.json."]
    lines+=['',f'Historical user-reported BioMistral: 9/58 = 15.52%, Wilson 95% {wilson(9,58)}. Not rerun; existing prompt-development runs differ, so this is not a controlled new comparison.',f'Human unanimous agreement: 38/58 = 65.52%, Wilson 95% {wilson(38,58)}. Majority-only: 20/58 = 34.48%, Wilson 95% {wilson(20,58)}. These are agreement proportions, not model accuracies, and unanimity is not an accuracy ceiling.','', '## Failures and limits','', '- Initial background launcher produced no output; relaunched with a tracked process and logs.', '- Baseline dataset preparation could not write the shared Hugging Face lock under sandbox permissions. Read the existing Arrow cache directly, without modifying it. Baseline inference was not rerun.', '- Severity initial alignment check failed on reversible UTF-8/Windows-1252 mojibake; logged and repaired in memory, then restarted before any CV result.', '- First detector launch failed before training on Windows default text decoding; UTF-8 reads fixed. Original failure logs retained.', '- Severity candidates did not show convincing improvement over majority. No severity transformer fine-tuning was justified.', '- Historical evidence-insensitivity and pharmacy domain-shift findings remain limitations, not problems solved by a high aggregate score.','', '## Reproduction','', 'From the project directory, with cached dependencies and models:', '```powershell', '.\.venv\Scripts\python.exe honest_baselines_v1.py', '.\.venv\Scripts\python.exe honest_severity_v1.py', '.\.venv\Scripts\python.exe honest_detection_v1.py', '.\.venv\Scripts\python.exe honest_final_evaluation_v1.py', '.\.venv\Scripts\python.exe honest_report_v1.py', '.\.venv\Scripts\python.exe honest_verify_v1.py', '.\.venv\Scripts\python.exe -m streamlit run app_v2.py', '```', 'The final evaluator refuses a second pass in this run folder. For independent reproduction, use a fresh checkout/output folder and preserve this run. No test-driven tuning is permitted. Seeds 42,43,44; split seed 42. Protocol and complete metrics: runs/honest_v1/.','', '## Claims I can honestly make','', '- All three historical detector counts were reproduced, with small-sample uncertainty explicitly reported.', '- Historical validation has question leakage; new training uses grouped exclusions and dev-only selection.', '- Severity labels align, and learned candidates were evaluated with nested question-grouped CV.', '- The keyword severity baseline is reproduced. Nested learned severity did not beat the majority baseline.', '- These experiments measure agreement with dataset labels, not clinical safety or reliable evidence verification.','', '## Claims I should not make','', '- “HalluGuard is 95% correct for pharmacy questions.”', '- “The original 94.7% proves generalization to unseen questions.”', '- “A supported prediction proves that the evidence supports the claim.”', '- “Within-one-tier accuracy is exact severity accuracy.”', '- “65.5% unanimity is a mathematical ceiling on consensus accuracy.”', '- “A score on 58 claims validates clinical use.”', '']
    lines += ['## Integration and file manifest','', 'The app uses the dev-locked detector when available, with fixed threshold 0.5 and dev-fitted temperature. It supports pasted answers without Ollama. Missing spaCy language data falls back to sentence splitting. Severity tier is conditional consequence; priority scales expected ordinal tier by detector probability. Missing severity artifact falls back to the original keyword function.','', 'Created scripts: honest_common_v1.py, honest_baselines_v1.py, honest_detection_v1.py, honest_final_evaluation_v1.py, honest_historical_diagnostics_v1.py, honest_severity_v1.py, honest_report_v1.py, honest_verify_v1.py, honest_finalize_v1.py, severity_features_v1.py, severity_scorer_v2.py, detector_honest_v1.py. Modified: app_v2.py. Outputs: runs/results_summary.md and all files under runs/honest_v1/. Detailed recursive artifact list and hashes are in files_manifest.json when final verification completes. Python may also create __pycache__ entries.','', 'Integration failures: original app import failed because en_core_web_sm is not installed; an offline sentence-splitting fallback was added without changing the extractor. Sandboxed Streamlit tests also hung in Windows socketpair setup before app execution; the same tests passed outside the sandbox. Final verification: six tests passed, including protected-file hashes, train/dev/test exclusions, nested group isolation, severity API/fallback, Streamlit initial render, and real-model pasted-answer analysis with Ollama calls prohibited. Logs: verification_final.log.','', 'Additional reproduction commands (after training and final evaluation):','```powershell','.\\.venv\\Scripts\\python.exe honest_historical_diagnostics_v1.py','.\\.venv\\Scripts\\python.exe honest_finalize_v1.py','```','Training took approximately 67 minutes, within the five-hour budget. Initial pilot estimate was 157 minutes; exact per-job durations, hardware and dependency versions are saved. Models may truncate long inputs using longest-first truncation at the dev-selected length.','']
    (ROOT/'runs/results_summary.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':main()
