"""Evidence-aware RoBERTa training. Use --dry-run to audit the grouped split."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')


def normalized(text):
    return ' '.join(text.split()).casefold()


def prepare_split(data, seed=42, val_fraction=0.2):
    """Group connected identical contexts/claims; deduplicate identical pairs."""
    rows, seen = [], {}
    for split in ('train', 'val'):
        for row in data[split]:
            if not all(isinstance(row.get(k), str) and row[k].strip()
                       for k in ('claim', 'context')) or row.get('label') not in (0, 1):
                raise ValueError('Rows need nonempty claim/context and label 0 or 1.')
            key = (normalized(row['context']), normalized(row['claim']))
            if key in seen:
                if seen[key] != row['label']:
                    raise ValueError('Conflicting labels for an identical pair.')
                continue
            seen[key] = row['label']
            rows.append({k: row[k] for k in ('claim', 'context', 'label')})
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, row in enumerate(rows):
        for field in ('context', 'claim'):
            key = (field, normalized(row[field]))
            if key in owners:
                parents[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    groups = list(groups.values())
    if len(groups) < 2:
        raise ValueError('Need at least two independent context/claim groups.')
    random.Random(seed).shuffle(groups)
    target = max(1, round(len(rows) * val_fraction))
    val, train = [], []
    for i, group in enumerate(groups):
        (val if len(val) < target and i < len(groups)-1 else train).extend(group)
    if any({r['label'] for r in split} != {0, 1} for split in (train, val)):
        raise ValueError('Both classes must occur in train and validation.')
    return train, val


def add_mismatch_negatives(rows, ratio=0.5, seed=42):
    """Add synthetic unsupported pairs within a split, excluding known supports.

    Different evidence can still support a claim: review saved synthetic pairs.
    These are not verified contradictions. Ratio counts negatives per positive.
    """
    rng = random.Random(seed)
    contexts = list(dict.fromkeys(r['context'] for r in rows))
    positives = [r for r in rows if r['label'] == 1]
    rng.shuffle(positives)
    supports = defaultdict(set)
    existing = {(normalized(r['claim']), normalized(r['context'])) for r in rows}
    for row in positives:
        supports[normalized(row['claim'])].add(normalized(row['context']))
    extra = []
    for row in positives[:round(len(positives) * ratio)]:
        claim = normalized(row['claim'])
        candidates = [c for c in contexts if normalized(c) not in supports[claim]
                      and (claim, normalized(c)) not in existing]
        if candidates:
            context = rng.choice(candidates)
            extra.append({'claim': row['claim'], 'context': context, 'label': 0})
            existing.add((claim, normalized(context)))
    return rows + extra


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='train_val_pairs_v3.json')
    parser.add_argument('--model', default='roberta-base',
                        help='Use a base model, not weights trained on the old overlapping split.')
    parser.add_argument('--output-dir', default=None, help='New empty run directory.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--gradient-accumulation', type=int, default=4)
    parser.add_argument('--negative-ratio', type=float, default=0.5)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--unweighted-loss', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.negative_ratio <= 1 or not 0 < args.val_fraction < 1:
        parser.error('negative-ratio must be in [0,1]; val-fraction must be in (0,1).')
    if min(args.epochs, args.batch_size, args.gradient_accumulation) < 1 or args.learning_rate <= 0:
        parser.error('Epochs, batch sizes and learning rate must be positive.')
    if not 8 <= args.max_length <= 512:
        parser.error('max-length must be between 8 and 512.')
    return args


def main():
    args = parse_args()
    data = json.loads(Path(args.data).read_text(encoding='utf-8'))
    train, val = prepare_split(data, args.seed, args.val_fraction)
    train_aug = add_mismatch_negatives(train, args.negative_ratio, args.seed)
    val_aug = add_mismatch_negatives(val, 1.0, args.seed + 1)
    counts = Counter(r['label'] for r in train_aug)
    weights = [len(train_aug) / (2 * counts[c]) for c in (0, 1)]
    if args.unweighted_loss:
        weights = [1.0, 1.0]
    print(f'Grouped split: train={len(train)}, validation={len(val)}')
    print(f'Training after augmentation: {len(train_aug)}; labels={dict(counts)}')
    print(f'Class weights: {weights}; validation mismatches={len(val_aug)-len(val)}')
    print('Old reported accuracy is not directly comparable to this context-held-out split.')
    if args.dry_run:
        return

    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, EarlyStoppingCallback,
                              Trainer, TrainingArguments, set_seed)

    set_seed(args.seed)
    output = Path(args.output_dir or
                  f'training_runs/evidence-v3-{datetime.now():%Y%m%d-%H%M%S-%f}')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)

    def save_json(name, value):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

    save_json('split.json', {'train': train, 'val': val})
    save_json('synthetic_negatives.json', {
        'train': train_aug[len(train):], 'val': val_aug[len(val):]})
    save_json('run_config.json', {**vars(args), 'class_weights': weights,
                                  'selection_metric': 'original_macro_f1'})
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    def to_dataset(rows):
        for i, row in enumerate(rows):
            if len(tokenizer.encode(row['claim'], add_special_tokens=False)) + 4 >= args.max_length:
                raise ValueError(f'Claim {i} exceeds token budget; curate/split before training.')
        ds = Dataset.from_list(rows)
        return ds.map(lambda b: tokenizer(b['context'], b['claim'],
                                         truncation='only_first', max_length=args.max_length),
                      batched=True, remove_columns=['claim', 'context']).rename_column('label', 'labels')

    train_ds, val_ds, val_aug_ds = map(to_dataset, (train_aug, val, val_aug))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, local_files_only=True,
        id2label={0: 'unsupported', 1: 'supported'},
        label2id={'unsupported': 0, 'supported': 1})

    class BalancedTrainer(Trainer):
        def __init__(self, *positional, **kwargs):
            super().__init__(*positional, **kwargs)
            # Loss is a microbatch mean; Trainer handles accumulation scaling.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs['labels']
            outputs = model(**{k: v for k, v in inputs.items() if k != 'labels'})
            logits = outputs.logits.float()
            losses = F.cross_entropy(logits, labels, reduction='none')
            loss = (losses * logits.new_tensor(weights)[labels]).mean()
            return (loss, outputs) if return_outputs else loss

    def metrics(pred):
        predicted = np.argmax(pred.predictions, axis=1)
        p, r, f1, _ = precision_recall_fscore_support(
            pred.label_ids, predicted, labels=[0, 1], zero_division=0)
        return {'accuracy': accuracy_score(pred.label_ids, predicted),
                'balanced_accuracy': balanced_accuracy_score(pred.label_ids, predicted),
                'macro_f1': float(f1.mean()), 'supported_precision': float(p[1]),
                'supported_recall': float(r[1]), 'unsupported_recall': float(r[0])}

    class OriginalValidationEarlyStopping(EarlyStoppingCallback):
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            # Trainer calls callbacks separately for each named validation set.
            # Only the original examples should update the patience counter.
            if args.metric_for_best_model in metrics:
                return super().on_evaluate(args, state, control, metrics, **kwargs)

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    steps_per_epoch = math.ceil(math.ceil(len(train_aug) / args.batch_size) / args.gradient_accumulation)
    training_args = TrainingArguments(
        output_dir=str(output / 'checkpoints'),
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        weight_decay=0.01, warmup_steps=max(1, math.ceil(0.1 * steps_per_epoch * args.epochs)),
        lr_scheduler_type='linear', max_grad_norm=1.0,
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='eval_original_macro_f1',
        greater_is_better=True, seed=args.seed, data_seed=args.seed,
        bf16=bf16, fp16=cuda and not bf16, dataloader_pin_memory=cuda,
        logging_steps=10, report_to='none')
    trainer = BalancedTrainer(
        model=model, args=training_args, train_dataset=train_ds,
        eval_dataset={'original': val_ds, 'with_mismatches': val_aug_ds},
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics,
        callbacks=[OriginalValidationEarlyStopping(early_stopping_patience=2)])
    print(f'Training device: {trainer.args.device}; output: {output.resolve()}')
    trainer.train()
    results = trainer.evaluate()
    results['best_checkpoint'] = trainer.state.best_model_checkpoint
    save_json('validation_metrics.json', results)
    predictions = trainer.predict(val_ds)
    scores = torch.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].tolist()
    save_json('validation_predictions.json', [
        {**row, 'probability_supported': score, 'predicted_label': int(score >= 0.5)}
        for row, score in zip(val, scores)])
    trainer.save_model(str(output / 'model'))
    tokenizer.save_pretrained(str(output / 'model'))
    print(f"Saved best model: {(output / 'model').resolve()}")
    print('Evaluate on an untouched test set before changing the app model path.')


if __name__ == '__main__':
    main()
