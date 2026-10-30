"""Run the frozen 80-fit CPU matrix; each fit has an independent output directory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--config', type=Path, default=Path('configs/scaling.json'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    for n in cfg['images']:
        for seed in cfg['seeds']:
            for kind in cfg['kinds']:
                name = f'{kind}-{n}-{seed}'
                run = dict(images=n, seed=seed, kind=kind, objective='bce',
                           updates=cfg['updates'], selection=cfg['selection'])
                path = args.output / (name + '.json')
                path.write_text(json.dumps(run, indent=2) + '\n')
                subprocess.run([sys.executable, '-m', 'training.readouts', '--features', args.features,
                                '--config', str(path), '--output', str(args.output / name)], check=True)


if __name__ == '__main__':
    main()
