"""Single-image/single-caption or resumable CSV scoring using frozen assets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--model', default='Linear-1024')
    ap.add_argument('--image')
    ap.add_argument('--caption')
    ap.add_argument('--input-csv', type=Path, help='image_id,image_path,caption; optional image_sha256')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not args.input_csv and (not args.image or args.caption is None):
        ap.error('provide --input-csv or both --image and --caption')
    from scripts.gpu_admission import check_idle
    check_idle()
    import torch
    from inference.full_label import Engine
    from inference.readouts import load_readout, score
    registry = json.loads((args.assets / 'models.json').read_text())
    spec = registry['models'][args.model]
    arm = spec['adapter_arm']
    engine = Engine(args.base_model, str(args.assets / 'adapters' / arm))
    engine.model.requires_grad_(False)
    engine.capture_hidden()
    if spec['form'] != 'z':
        net, scale = load_readout(args.assets / spec['checkpoint'], args.assets / spec['standardizer'],
                                  'linear_h' if spec['form'] == 'linear' else 'residual_hz', 'cuda')
    rows = (list(csv.DictReader(args.input_csv.open())) if args.input_csv else
            [dict(image_id=Path(args.image).stem, image_path=args.image, caption=args.caption)])
    done = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        done = {json.loads(s).get('input_key') for s in args.output.read_text().splitlines()}
    try:
        with args.output.open('a') as stream:
            for row in rows:
                try:
                    image_hash = hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest()
                except FileNotFoundError:
                    stream.write(json.dumps(dict(image_id=row['image_id'], model=args.model,
                                                 status='INPUT_FAILURE', error='FileNotFoundError')) + '\n')
                    stream.flush()
                    continue
                registry_hash = hashlib.sha256((args.assets / 'models.json').read_bytes()).hexdigest()
                key = hashlib.sha256(('\0'.join([args.model, registry_hash, image_hash,
                                                 row['caption']])).encode()).hexdigest()
                if key in done:
                    continue
                if row.get('image_sha256'):
                    assert hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest() == row['image_sha256']
                try:
                    with torch.inference_mode():
                        inp = engine.inputs(row['image_path'], row['caption'], registry['prompt_template'])
                        h, z = engine.live(inp)
                        value = float(z if spec['form'] == 'z' else score(net, h, z, scale,
                                      'linear_h' if spec['form'] == 'linear' else 'residual_hz'))
                    result = dict(input_key=key, image_id=row['image_id'], model=args.model,
                                  score=value, threshold=spec['threshold'], status='SCORED',
                                  prediction='CONFLICT' if value > spec['threshold'] else 'COMPATIBLE')
                except (ValueError, FileNotFoundError) as exc:
                    result = dict(input_key=key, image_id=row['image_id'], model=args.model,
                                  status='INPUT_FAILURE', error=type(exc).__name__)
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                done.add(key)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
