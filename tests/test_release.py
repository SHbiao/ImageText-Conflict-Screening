"""CPU-only numerical and checkpoint checks; no model generation or training."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import metrics, frozen_threshold, pair_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    import torch
    from sklearn.metrics import roc_curve, roc_auc_score
    from safetensors import safe_open
    from inference.readouts import load_readout, score
    torch.set_num_threads(4)
    checks = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        y = np.tile([0, 1], 100)
        s = rng.integers(-10, 11, len(y)) / 3  # include repeated scores
        got = metrics(y, s, .5)
        f, t, _ = roc_curve(y, s)
        assert abs(got['AUROC'] - roc_auc_score(y, s)) < 1e-12
        for beta, name in [(.05, 'pAUC05'), (.1, 'pAUC10')]:
            ix = f < beta
            expected = np.trapz(np.r_[t[ix], np.interp(beta, f, t)], np.r_[f[ix], beta]) / beta
            assert abs(got[name] - expected) < 1e-12
    assert frozen_threshold([0] * 20 + [1], np.arange(21)) == 18
    assert metrics([0, 1], [1., 1.], 1.)['FP'] == 0
    assert pair_metrics(['a', 'a', 'b', 'b'], [0, 1, 0, 0], [0, 1, 1, 2])['pair_images'] == 1
    checks.append('30 tied-score ROC/pAUC fixtures against scikit-learn; strict threshold; qualified pairs')
    registry = json.loads((args.assets / 'models.json').read_text())
    for arm, files in registry['adapters'].items():
        for name, expected in files.items():
            assert hashlib.sha256((args.assets / 'adapters' / arm / name).read_bytes()).hexdigest() == expected
        with safe_open(args.assets / 'adapters' / arm / 'adapter_model.safetensors', framework='pt', device='cpu') as state:
            assert len(state.keys()) > 0
            for key in state.keys():
                assert torch.isfinite(state.get_tensor(key)).all()
    checks.append('both adapter files readable on CPU; all tensors finite; registry hashes match')
    predictions = list(csv.DictReader((args.data / 'predictions/readout_scores/human_predictions.csv').open()))
    errors = {}
    for name, spec in registry['models'].items():
        arm = spec['adapter_arm']
        h = torch.from_numpy(np.load(args.data / f'features/human_{arm}_h.npy'))
        rows = list(csv.DictReader((args.data / f'features/human_{arm}_rows.csv').open()))
        z = torch.tensor([float(r['z']) for r in rows])
        if spec['form'] == 'z':
            values = z.numpy()
        else:
            for field in ['checkpoint', 'standardizer']:
                assert hashlib.sha256((args.assets / spec[field]).read_bytes()).hexdigest() == spec[field + '_sha256']
            kind = 'linear_h' if spec['form'] == 'linear' else 'residual_hz'
            net, scale = load_readout(args.assets / spec['checkpoint'], args.assets / spec['standardizer'], kind)
            with torch.inference_mode():
                values = score(net, h, z, scale, kind).numpy()
        actual = {r['case_id']: float(values[i]) for i, r in enumerate(rows)}
        truth = [r for r in predictions if r['model'] == name]
        assert len(truth) == 1050
        error = max(abs(actual[r['case_id']] - float(r['score'])) for r in truth)
        assert error < 1e-5, (name, error)
        assert all((actual[r['case_id']] > spec['threshold']) == (r['prediction'] == 'CONFLICT') for r in truth)
        errors[name] = error
    for source in Path(__file__).resolve().parents[1].rglob('*.py'):
        ast.parse(source.read_text())
    result = dict(status='PASSED', checks=checks, replayed_conditions=6300,
                  max_absolute_score_errors=errors, GPU_calls=0, training_runs=0,
                  GPU_portability='source inspected; no new GPU inference/training run')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
