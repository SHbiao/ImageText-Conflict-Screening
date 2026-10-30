"""Live, differentiable two-label QLoRA + residual readout adaptation."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True, help='JSON image groups: split,image_path,captions')
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--image-root', type=Path, required=True)
    args = ap.parse_args()
    from scripts.gpu_admission import check_idle
    check_idle(min_free_mib=14000)
    import torch
    from inference.full_label import Engine
    from inference.readouts import load_readout, score
    from evaluation.metrics import metrics, frozen_threshold
    cfg = json.loads(args.config.read_text())
    torch.manual_seed(cfg['seed'])
    registry = json.loads((args.assets / 'models.json').read_text())
    spec = registry['models']['Residual-1536']
    engine = Engine(args.base_model, str(args.assets / 'adapters/original'), trainable=True)
    model = engine.model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert named and all('lora_' in n and ('.q_proj.' in n or '.v_proj.' in n)
                         and 'visual' not in n for n, p in named)
    net, scale = load_readout(args.assets / spec['checkpoint'], args.assets / spec['standardizer'],
                              'residual_hz', 'cuda')
    net.requires_grad_(True)
    engine.capture_hidden()
    groups = json.loads(args.manifest.read_text())
    for g in groups:
        g['image_path'] = g.get('image_path') or str(args.image_root / g['image_file'])
    train = [g for g in groups if g['split'] == 'train']
    dev = [g for g in groups if g['split'] == 'dev']
    assert len(train) == 1536 and len(dev) == 256
    assert not ({g['image_id'] for g in train} & {g['image_id'] for g in dev})
    args.output.mkdir(parents=True, exist_ok=False)
    params = [p for n, p in named]
    opt = torch.optim.AdamW([dict(params=params, lr=cfg['lora_lr']),
                            dict(params=net.parameters(), lr=cfg['head_lr'])],
                           weight_decay=cfg['weight_decay'])
    opt.zero_grad(set_to_none=True)
    seen = steps = 0
    log = []
    try:
        for epoch in range(3):
            model.train()
            net.train()
            for index in np.random.default_rng(cfg['seed'] + epoch).permutation(len(train)):
                g = train[index]
                for condition in ['normal', 'conflict']:
                    inp = engine.inputs(g['image_path'], g['captions'][condition], registry['prompt_template'])
                    h, z = engine.live(inp)
                    assert h.requires_grad and z.requires_grad
                    value = score(net, h, z, scale, 'residual_hz')
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(value,
                                torch.tensor(float(condition == 'conflict'), device='cuda'))
                    (loss / 32).backward()
                    seen += 1
                    engine.capture.clear()
                    if seen % 32 == 0:
                        assert any(p.grad is not None and p.grad.abs().max() > 0 for p in params)
                        torch.nn.utils.clip_grad_norm_(params + list(net.parameters()), cfg['gradient_clip'])
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        steps += 1
                        if steps % cfg['evaluate_every_steps'] == 0:
                            model.eval()
                            net.eval()
                            values = []
                            with torch.inference_mode():
                                for gdev in dev:
                                    for c in ['normal', 'conflict']:
                                        h, z = engine.live(engine.inputs(gdev['image_path'], gdev['captions'][c],
                                                                        registry['prompt_template']))
                                        values.append(float(score(net, h, z, scale, 'residual_hz')))
                            labels = np.tile([0, 1], len(dev))
                            threshold = frozen_threshold(labels, values)
                            checkpoint = args.output / f'step{steps:03d}'
                            model.save_pretrained(checkpoint, safe_serialization=True)
                            torch.save(net.state_dict(), checkpoint / 'readout.pt')
                            log.append(dict(step=steps, conditions=seen, threshold=threshold,
                                            metrics=metrics(labels, values, threshold)))
                            np.save(checkpoint / 'dev-scores.npy', values)
                            (args.output / 'history.json').write_text(json.dumps(log, indent=2))
                            model.train()
                            net.train()
        assert (seen, steps) == (9216, 288)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
