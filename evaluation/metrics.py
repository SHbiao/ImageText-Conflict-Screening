"""Integral-normalized ROC metrics and image-group uncertainty used in the paper."""
from collections import defaultdict
import numpy as np


def metrics(labels, scores, threshold):
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != s.shape or not len(y) or not np.isfinite(s).all():
        raise ValueError('Expected equal-length, finite binary labels and scores')
    if not set(y) <= {0, 1}:
        raise ValueError('Binary metrics require labels 0 and 1')
    pred = s > threshold
    pos, neg = y == 1, y == 0
    out = dict(N=len(y), N_conflict=int(pos.sum()), N_compatible=int(neg.sum()),
               TP=int((pred & pos).sum()), FN=int((~pred & pos).sum()),
               FP=int((pred & neg).sum()), TN=int((~pred & neg).sum()),
               Accuracy=float((pred == y).mean()),
               Recall=float(pred[pos].mean()) if pos.any() else None,
               FAR=float(pred[neg].mean()) if neg.any() else None,
               Retention=float((~pred[neg]).mean()) if neg.any() else None,
               AUROC=None, pAUC05=None, pAUC10=None, ROC_Recall5=None)
    if not (pos.any() and neg.any()):
        return out
    order = np.argsort(-s, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(s[order])), len(s) - 1]
    t = np.r_[0., np.cumsum(y[order])[ends] / pos.sum()]
    f = np.r_[0., np.cumsum(1 - y[order])[ends] / neg.sum()]
    out.update(AUROC=float(np.trapz(t, f)), ROC_Recall5=float(t[f <= .05].max()))
    for beta, name in [(.05, 'pAUC05'), (.1, 'pAUC10')]:
        ix = f < beta
        out[name] = float(np.trapz(np.r_[t[ix], np.interp(beta, f, t)],
                                   np.r_[f[ix], beta]) / beta)
    return out


def frozen_threshold(labels, scores):
    normal = np.sort(np.asarray(scores)[np.asarray(labels) == 0])[::-1]
    if not len(normal):
        raise ValueError('Calibration needs compatible development inputs')
    return float(normal[int(np.floor(.05 * len(normal)))])


def pair_metrics(image_ids, labels, scores):
    groups = defaultdict(list)
    for image, y, s in zip(image_ids, labels, scores):
        groups[str(image)].append((int(y), float(s)))
    gaps = []
    for values in groups.values():
        if len(values) == 2 and {v[0] for v in values} == {0, 1}:
            gaps.append(sum(s if y else -s for y, s in values))
    return dict(pair_images=len(gaps), Pair_Accuracy=float(np.mean(np.array(gaps) > 0))
                if gaps else None, Pair_Gap_mean=float(np.mean(gaps)) if gaps else None)


def paired_bootstrap(image_ids, labels, a, b, threshold_a, threshold_b,
                     repetitions=2000, seed=2026102911, alpha=.05):
    groups = defaultdict(list)
    for i, image in enumerate(image_ids):
        groups[str(image)].append(i)
    blocks = list(groups.values())
    y, a, b = map(np.asarray, (labels, a, b))
    rng = np.random.default_rng(seed)
    keys = ['pAUC10', 'ROC_Recall5', 'Recall', 'FAR']
    traces = {k: [] for k in keys}
    for _ in range(repetitions):
        draw = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        ma, mb = metrics(y[draw], a[draw], threshold_a), metrics(y[draw], b[draw], threshold_b)
        for k in keys:
            if ma[k] is not None and mb[k] is not None:
                traces[k].append(ma[k] - mb[k])
    return {k: dict(CI_low=float(np.quantile(v, alpha / 2)),
                    CI_high=float(np.quantile(v, 1 - alpha / 2)), valid_draws=len(v))
            for k, v in traces.items() if v}
