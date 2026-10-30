"""Reconstruct the five-seed architecture-by-supervision matrix."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from evaluation.data import read, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    groups = defaultdict(list)
    for row in read(args.metrics):
        groups[row['kind'], row['images'], row['split']].append(row)
    out = []
    for (kind, images, split), rows in groups.items():
        assert len(rows) == 5
        result = dict(kind=kind, images=images, split=split, seeds=5)
        for k in ['pAUC10', 'pAUC05', 'AUROC', 'recall', 'FAR', 'pair_accuracy']:
            values = [float(r[k]) for r in rows]
            result[k + '_mean'] = float(np.mean(values))
            result[k + '_sd'] = float(np.std(values, ddof=1))
        out.append(result)
    write(args.output, out)


if __name__ == '__main__':
    main()
