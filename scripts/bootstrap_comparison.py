"""Image-paired uncertainty for a pre-specified two-model score comparison."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.data import read
from evaluation.metrics import metrics, paired_bootstrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--model-a', required=True)
    ap.add_argument('--model-b', required=True)
    ap.add_argument('--repetitions', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--alpha', type=float, default=.05)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    table = read(args.predictions)
    by_model = {}
    for name in [args.model_a, args.model_b]:
        rows = [r for r in table if r['model'] == name]
        keys = [(r['image_id'], r['condition']) for r in rows]
        assert rows and len(set(keys)) == len(rows), 'use a single cohort/split/model'
        by_model[name] = dict(zip(keys, rows))
    a, b = by_model.values()
    assert a.keys() == b.keys(), 'different comparison denominators'
    keys = list(a)
    y = [int(a[k].get('label', a[k].get('source_label', k[1] == 'conflict'))) for k in keys]
    sa, sb = ([float(x[k]['score']) for k in keys] for x in [a, b])
    ta, tb = (float(next(iter(x.values()))['threshold']) for x in [a, b])
    ma, mb = metrics(y, sa, ta), metrics(y, sb, tb)
    interval = paired_bootstrap([k[0] for k in keys], y, sa, sb, ta, tb,
                               args.repetitions, args.seed, args.alpha)
    result = dict(model_a=args.model_a, model_b=args.model_b, conditions=len(keys),
        images=len({k[0] for k in keys}), repetitions=args.repetitions, seed=args.seed,
        interval_level=1-args.alpha, thresholds_refit=False,
        differences={k:dict(point=ma[k]-mb[k], **v) for k,v in interval.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
