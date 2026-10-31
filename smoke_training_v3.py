"""Exercise the real Trainer/save/reload path with tiny random weights, not accuracy.

Uses the existing local tokenizer; all temporary outputs are removed on exit.
"""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch
from transformers import AutoModelForSequenceClassification, RobertaConfig, RobertaForSequenceClassification

from fiinetune_roberta_v3 import main
from test_training_v3 import fixture


def smoke_test():
    config = RobertaConfig(vocab_size=50265, hidden_size=16, num_hidden_layers=1,
                           num_attention_heads=2, intermediate_size=32,
                           max_position_embeddings=514, num_labels=2)
    with tempfile.TemporaryDirectory(prefix='v3-smoke-', dir='.') as directory:
        root = Path(directory)
        data = root / 'pairs.json'
        data.write_text(json.dumps(fixture()), encoding='utf-8')
        output = root / 'run'
        argv = ['fiinetune_roberta_v3.py', '--data', str(data), '--model',
                'roberta-halluguard-final', '--output-dir', str(output),
                '--epochs', '1', '--batch-size', '2', '--gradient-accumulation', '2',
                '--max-length', '32']
        with patch.object(sys, 'argv', argv), patch.object(
                AutoModelForSequenceClassification, 'from_pretrained',
                return_value=RobertaForSequenceClassification(config)):
            main()
        metrics = json.loads((output / 'validation_metrics.json').read_text())
        assert 'eval_original_macro_f1' in metrics
        assert 'eval_with_mismatches_accuracy' in metrics
        assert metrics['best_checkpoint']
        reloaded = AutoModelForSequenceClassification.from_pretrained(output / 'model')
        assert all(torch.isfinite(p).all() for p in reloaded.parameters())
        print('PASS: real training, both evaluations, best-checkpoint loading and export/reload.')


if __name__ == '__main__':
    smoke_test()
