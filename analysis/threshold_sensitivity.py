"""Reproduce fixed-model calibration resampling without fitting to evaluation labels."""
import argparse
from pathlib import Path
import numpy as np
from evaluation.data import read, write
from evaluation.metrics import frozen_threshold, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=2026102820)
    ap.add_argument('--repetitions', type=int, default=1000)
    args = ap.parse_args()
    rows = [r for r in read(args.predictions) if r['model'] == args.model]
    normal = np.array([float(r['score']) for r in rows if r['split'] == 'dev' and r['condition'] == 'normal'])
    test = [r for r in rows if r['split'] == 'test']
    assert len(normal) == 256
    y = [r['condition'] == 'conflict' for r in test]
    s = [float(r['score']) for r in test]
    rng = np.random.default_rng(args.seed)
    out = []
    for size in [32, 64, 128, 256]:
        for trial in range(args.repetitions):
            calibration = rng.choice(normal, size, replace=False)
            threshold = frozen_threshold(np.zeros(size), calibration)
            m = metrics(y, s, threshold)
            out.append(dict(calibration_images=size, trial=trial, threshold=threshold,
                            Recall=m['Recall'], FAR=m['FAR']))
    write(args.output, out)


if __name__ == '__main__':
    main()
