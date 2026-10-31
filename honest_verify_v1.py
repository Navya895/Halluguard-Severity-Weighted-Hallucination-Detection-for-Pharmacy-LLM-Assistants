"""Integrity and offline integration checks; no reevaluation of test predictions."""
from honest_common_v1 import *
import unittest,importlib,sys
from unittest.mock import patch
class IntegrityTests(unittest.TestCase):
    def test_protected_files(self):
        hashes=json.loads((OUT/'protected_hashes.json').read_text(encoding='utf-8'))
        for name,sha in hashes.items():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),sha,name)
    def test_split_exclusion(self):
        splits=json.loads((OUT/'detection_split.json').read_text(encoding='utf-8'));forbidden=json.loads((OUT/'pharmacy_exclusions.json').read_text(encoding='utf-8'))
        tr={norm(r['Question']) for r in splits['train']};dv={norm(r['Question']) for r in splits['dev']}
        self.assertFalse(tr&dv);self.assertFalse((tr|dv)&set(forbidden['questions']))
        for r in splits['train']+splits['dev']:
            for k in ['Ground Truth','Hallucinated Answer']:self.assertNotIn(norm(r[k]),forbidden['claims'])
    def test_severity_api_and_fallback(self):
        import severity_scorer_v2 as s
        from severity_scorer import compute_severity as old
        text='An invented dosage statement for interface testing.'
        result=s.compute_severity(.7,text,.2)
        self.assertEqual(set(result),set(old(.7,text)));self.assertIn(result['tier'],s.TIERS)
        with patch.object(s,'MODEL_PATH',OUT/'missing-model-for-test.joblib'):
            self.assertEqual(s.compute_severity(.7,text,.2),old(.7,text))
        for p in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):s.compute_severity(p,text)
    def test_cv_groups(self):
        folds=json.loads((OUT/'severity_folds.json').read_text(encoding='utf-8'));groups=json.loads((OUT/'severity_predictions.json').read_text(encoding='utf-8'))['groups']
        for f in folds:
            self.assertFalse({groups[i] for i in f['train_ids']}&{groups[i] for i in f['test_ids']})
            for a,b in f['inner_splits']:
                self.assertFalse({groups[i] for i in a}&{groups[i] for i in b})
                self.assertTrue(set(a+b)<=set(f['train_ids']))
    def test_app_initial_render(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(ROOT/'app_v2.py')).run(timeout=90)
        self.assertEqual(len(app.exception),0,str(app.exception))
    def test_app_pasted_answer_without_ollama(self):
        if not (OUT/'selection_LOCKED.json').exists():self.skipTest('Detector training has not finished')
        from streamlit.testing.v1 import AppTest
        with patch('llm_connector.query_biomistral',side_effect=AssertionError('Ollama must not be called')):
            app=AppTest.from_file(str(ROOT/'app_v2.py')).run(timeout=90)
            app.text_area(key='question_input').set_value('What does this demonstration say?')
            app.text_area(key='reference_input').set_value('This is a software demonstration.')
            app.text_area(key='pasted_answer_input').set_value('This is a software demonstration.')
            app.button[0].click().run(timeout=120)
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertFalse(any('SCORING FAILED' in str(x.value) for x in app.markdown))
            self.assertTrue(any('Claim-Level' in str(x.value) for x in app.subheader))
if __name__=='__main__':unittest.main()
