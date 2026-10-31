"""Data-integrity regression tests; no model downloads or training required."""
import unittest

from fiinetune_roberta_v3 import add_mismatch_negatives, normalized, prepare_split


def fixture():
    rows = []
    for i in range(10):
        rows.extend([
            {'context': f'evidence {i}', 'claim': f'true {i}', 'label': 1},
            {'context': f'evidence {i}', 'claim': f'false {i}', 'label': 0},
        ])
    return {'train': rows[::2], 'val': rows[1::2]}


class TrainingDataTests(unittest.TestCase):
    def test_contexts_and_claims_do_not_leak(self):
        data = fixture()
        # A shared claim connects two otherwise different evidence groups.
        data['train'][1]['claim'] = data['train'][0]['claim']
        train, val = prepare_split(data)
        for field in ('context', 'claim'):
            self.assertFalse({normalized(r[field]) for r in train}
                             & {normalized(r[field]) for r in val})
        self.assertEqual(len(train) + len(val), 20)
        self.assertEqual((train, val), prepare_split(data))

    def test_conflicting_labels_are_rejected(self):
        data = fixture()
        data['val'].append({**data['train'][0], 'label': 0})
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            prepare_split(data)

    def test_duplicate_pairs_are_removed(self):
        data = fixture()
        data['val'].append(dict(data['train'][0]))
        train, val = prepare_split(data)
        self.assertEqual(len(train) + len(val), 20)

    def test_mismatches_stay_local_and_avoid_known_support(self):
        train, _ = prepare_split(fixture())
        augmented = add_mismatch_negatives(train, 1.0)
        supports = {(r['claim'], r['context']) for r in train if r['label'] == 1}
        self.assertGreater(len(augmented), len(train))
        for row in augmented[len(train):]:
            self.assertEqual(row['label'], 0)
            self.assertIn(row['context'], {r['context'] for r in train})
            self.assertNotIn((row['claim'], row['context']), supports)
        self.assertEqual(augmented, add_mismatch_negatives(train, 1.0))

    def test_no_candidate_terminates_without_false_negative(self):
        rows = [{'claim': 'same claim', 'context': c, 'label': 1}
                for c in ('evidence A', 'evidence B')]
        self.assertEqual(add_mismatch_negatives(rows, 1.0), rows)
        self.assertEqual(add_mismatch_negatives(rows, 0), rows)


if __name__ == '__main__':
    unittest.main()
