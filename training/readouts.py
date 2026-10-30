"""CPU port of the retained fixed-update and matched-objective head trainers."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from evaluation.metrics import metrics, frozen_threshold
from inference.readouts import network, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    rows = list(csv.DictReader((args.features / 'readout_rows.csv').open()))
    h = np.load(args.features / 'readout_h.npy')
    z = np.array([float(r['z']) for r in rows], dtype=np.float32)
    pool = np.array([i for i, r in enumerate(rows) if r['split'] in ['train', 'extra_train']]).reshape(-1, 2)
    dev = np.array([i for i, r in enumerate(rows) if r['split'] == 'dev'])
    for pair in pool:
        a, b = [rows[i] for i in pair]
        assert a['image_id'] == b['image_id'] and (a['condition'], b['condition']) == ('normal', 'conflict')
    original = pool[:1024]
    n = cfg['images']
    if n not in [256, 512, 1024, 1536]:
        raise ValueError('Published supervision scales are 256, 512, 1024, 1536 image groups')
    train = (original[np.sort(np.random.default_rng(cfg['seed']).permutation(1024)[:n])]
             if n < 1024 else original if n == 1024 else pool)
    assert not ({rows[i]['image_id'] for i in train.ravel()} & {rows[i]['image_id'] for i in dev})
    mu, sd = h[train.ravel()].mean(0), h[train.ravel()].std(0).clip(.001)
    zm, zs = z[train.ravel()].mean(), max(float(z[train.ravel()].std()), .001)
    scale = dict(mean=torch.tensor(mu), std=torch.tensor(sd),
                 z_mean=torch.tensor(zm), z_std=torch.tensor(zs))
    # Keep raw h; the common score function applies exactly one training-set standardization.
    tx, tz = torch.from_numpy(h[train]), torch.from_numpy(z[train])
    dx, dz = torch.from_numpy(h[dev]), torch.from_numpy(z[dev])
    labels = torch.tensor([0., 1.])
    dy = np.array([rows[i]['condition'] == 'conflict' for i in dev], dtype=int)
    kind, objective = cfg['kind'], cfg.get('objective', 'bce')
    if objective == 'anchor' and kind != 'residual_hz':
        raise ValueError('The matched anchor objective uses the residual u(h) interface')
    net = network(kind)
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    def evaluate():
        net.eval()
        with torch.inference_mode():
            s = score(net, dx, dz, scale, kind).numpy()
        return metrics(dy, s, frozen_threshold(dy, s))['pAUC10'], s
    best_metric, best_scores = evaluate()
    best = copy.deepcopy(net.state_dict())
    best_step = step = epoch = 0
    history = []
    start = time.monotonic()
    while step < cfg['updates']:
        epoch += 1
        perm = np.random.default_rng(cfg['seed'] + epoch).permutation(len(train))
        for begin in range(0, len(perm), 64):
            q = perm[begin:begin + 64]
            net.train()
            s = score(net, tx[q], tz[q], scale, kind)
            if objective == 'anchor':
                u = tz[q] - s
                loss = (u - tz[q].mean(1, keepdim=True)).square().mean()
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(s, labels.expand_as(s))
                if objective == 'ranking':
                    negatives = torch.topk(s[:, 0], max(1, math.ceil(.05 * len(q)))).values
                    loss = loss + .5 * torch.nn.functional.softplus(1 + negatives[None] - s[:, 1, None]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            opt.step()
            step += 1
            epoch_end = begin + 64 >= len(perm)
            check = ((step == 16 or step % 80 == 0) if cfg['selection'] == 'fixed_steps'
                     else (epoch_end and (epoch == 1 or epoch % 5 == 0)))
            if check:
                value, scores = evaluate()
                history.append(dict(step=step, epoch=epoch, loss=float(loss.detach()), dev_pAUC10=value))
                if value > best_metric + 1e-12:
                    best_metric, best_scores = value, scores.copy()
                    best, best_step = copy.deepcopy(net.state_dict()), step
            if step >= cfg['updates']:
                break
    torch.save(best, args.output / 'readout.pt')
    np.savez(args.output / 'standardizer.npz', mean=mu, std=sd, z_mean=zm, z_std=zs)
    result = dict(config=cfg, selected_step=best_step, dev_pAUC10=best_metric,
                  frozen_threshold=frozen_threshold(dy, best_scores), history=history,
                  CPU_seconds=time.monotonic() - start, GPU_seconds=0, test_used_for_selection=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
