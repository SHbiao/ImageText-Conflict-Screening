"""Sibling-dependent Oracle diagnosis on one unchanged, complete paired cohort."""
import argparse
import json
from collections import defaultdict
from evaluation.data import read
from evaluation.metrics import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--model', required=True)
    args = ap.parse_args()
    groups = defaultdict(dict)
    for r in read(args.predictions):
        if r['model'] == args.model:
            groups[r['image_id']][r['condition']] = float(r['score'])
    assert groups and all(set(p) == {'normal', 'conflict'} for p in groups.values())
    y, raw, centered = [], [], []
    for pair in groups.values():
        m = (pair['normal'] + pair['conflict']) / 2
        for condition in ['normal', 'conflict']:
            y.append(int(condition == 'conflict'))
            raw.append(pair[condition])
            centered.append(pair[condition] - m)
    print(json.dumps(dict(image_groups=len(groups), raw=metrics(y, raw, 0),
        sibling_oracle=metrics(y, centered, 0),
        interpretation='Sibling-dependent paired diagnostic; not deployable single-input performance'), indent=2))


if __name__ == '__main__':
    main()
