"""One CPU command reconstructs the core paper tables from released predictions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.data import read, write
from evaluation.metrics import metrics, pair_metrics
from evaluation.human import evaluate as human_metrics
from evaluation.sugarcrepe import evaluate as sugar_metrics


def grouped_scores(path):
    groups = defaultdict(list)
    for r in read(path):
        groups[r['model'], r.get('split', 'evaluation')].append(r)
    out = []
    for (model, split), rows in groups.items():
        planned = len(rows)
        valid = [r for r in rows if r.get('status', 'SCORED').startswith('SCORED')]
        labels = [int(r.get('label', r.get('source_label', r['condition'] == 'conflict'))) for r in valid]
        scores = [float(r['score']) for r in valid]
        thresholds = {float(r['threshold']) for r in valid}
        assert len(thresholds) == 1, ('mixed frozen thresholds', model, split)
        threshold = thresholds.pop()
        out.append(dict(model=model, split=split, planned_conditions=planned,
            failures=planned - len(valid), threshold=threshold, **metrics(labels, scores, threshold),
            **pair_metrics([r['image_id'] for r in valid], labels, scores)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('outputs/reproduced'))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks = []
    for file in ['matched_objectives.csv', 'simple_heads.csv', 'scale_confirmation.csv',
                 'adaptation_confirmation.csv', 'architecture_scaling.csv.gz']:
        rows = grouped_scores(args.data / 'predictions/readout_scores' / file)
        write(args.output / (file.replace('.csv.gz', '').replace('.csv', '') + '.csv'), rows)
        checks.append(dict(table=file, models=len(rows)))
    human = human_metrics(args.data)
    sugar = sugar_metrics(args.data)
    write(args.output / 'human_metrics.csv', human)
    write(args.output / 'sugarcrepe_metrics.csv', sugar)
    # Every archived point estimate is an assertion target, never copied as computed output.
    expected_h = {(r['model'], r['cohort'], r['scope'], r['labels']): r
                  for r in read(args.data / 'statistics/human_metrics.csv')}
    max_error = 0.
    for row in human:
        e = expected_h[row['model'], row['cohort'], row['scope'], row['labels']]
        for k in ['pAUC05', 'pAUC10', 'AUROC', 'Recall', 'FAR', 'Pair_Accuracy']:
            if e[k] and row[k] is not None:
                delta = abs(float(e[k]) - row[k])
                max_error = max(max_error, delta)
                assert delta < 1e-12, (row['model'], k, delta)
    expected_s = {(r['model'], r['category']): r for r in
                  read(args.data / 'statistics/sugarcrepe_metrics.csv')}
    for row in sugar:
        assert abs(row['native_accuracy_full_denominator'] - float(
            expected_s[row['model'], row['category']]['native_accuracy_full_denominator'])) < 1e-12
    record = dict(status='PASSED', reconstructed=checks, human_metric_rows=len(human),
                  sugarcrepe_metric_rows=len(sugar), max_human_numeric_error=max_error,
                  GPU_calls=0, training_runs=0)
    (args.output / 'validation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
