"""Separate within-image order, cross-image violations, and frozen decisions."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from evaluation.data import read, write
from evaluation.metrics import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    groups = defaultdict(list)
    for row in read(args.predictions):
        groups[row['model'], row['image_id']].append(row)
    negatives = defaultdict(list)
    for (model, image), rows in groups.items():
        for r in rows:
            if r['condition'] == 'normal':
                negatives[model].append((image, float(r['score'])))
    out = []
    for (model, image), rows in groups.items():
        if len(rows) != 2:
            continue
        normal = next(r for r in rows if r['condition'] == 'normal')
        edited = next(r for r in rows if r['condition'] == 'conflict')
        a, b = float(normal['score']), float(edited['score'])
        pool = np.array([s for i, s in negatives[model] if i != image])
        out.append(dict(model=model, image_id=image, pair_gap=b - a, pair_center=(a + b) / 2,
            pair_correct=b > a, edited_missed=b <= float(edited['threshold']),
            original_false_alarm=a > float(normal['threshold']),
            threshold_margin=b - float(edited['threshold']),
            cross_image_violation=float((pool >= b).mean()) if len(pool) else None))
    write(args.output, out)


if __name__ == '__main__':
    main()
