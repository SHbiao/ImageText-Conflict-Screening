"""Classification QLoRA: decision-value mean CE from the retained initialization."""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True, help='JSON list: image_path,caption,label,split')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--image-root', type=Path, required=True)
    args = ap.parse_args()
    from scripts.gpu_admission import check_idle
    check_idle(min_free_mib=14000)
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from inference.full_label import Engine
    cfg = json.loads(args.config.read_text())
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    rows = [r for r in json.loads(args.manifest.read_text()) if r['split'] == 'train']
    assert len(rows) == cfg['conditions']
    if not cfg.get('manifest_schedule_frozen', False):
        random.shuffle(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    engine = Engine(args.base_model)
    model = prepare_model_for_kbit_training(engine.model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={'use_reentrant': False})
    targets = [n for n, _ in model.named_modules() if 'language_model' in n and
               n.endswith(('self_attn.q_proj', 'self_attn.v_proj'))]
    assert targets and all('visual' not in n for n in targets)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                                           target_modules=targets, bias='none', task_type='CAUSAL_LM'))
    engine.model = model
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.)
    opt.zero_grad(set_to_none=True)
    log = []
    for i, row in enumerate(rows):
        image_path = row.get('image_path') or str(args.image_root / row['image_file'])
        inp = engine.inputs(image_path, row['caption'], cfg['prompt_template'])
        label = 'CONFLICT' if int(row['label']) else 'NO_CONFLICT'
        text = json.dumps(dict(decision=label, reason=''), separators=(',', ':'))
        enc = engine.processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        start = text.index(label)
        chosen = [j for j, (a, b) in enumerate(enc['offset_mapping']) if b > start and a < start + len(label)]
        n = inp['input_ids'].shape[1]
        out = dict(inp)
        out['input_ids'] = torch.cat([inp['input_ids'], torch.tensor([enc['input_ids']], device='cuda')], 1)
        out['attention_mask'] = torch.ones_like(out['input_ids'])
        if 'mm_token_type_ids' in out:
            out['mm_token_type_ids'] = torch.cat([inp['mm_token_type_ids'], torch.zeros(
                (1, len(enc['input_ids'])), device='cuda', dtype=inp['mm_token_type_ids'].dtype)], 1)
        logits = model(**out, use_cache=False).logits[0, [n + j - 1 for j in chosen]].float()
        targets = torch.tensor([enc['input_ids'][j] for j in chosen], device='cuda')
        loss = torch.nn.functional.cross_entropy(logits, targets, reduction='mean')
        (loss / 8).backward()
        log.append(dict(condition=i + 1, loss=float(loss.detach())))
        # Retain the original classification control's two no-gradient auxiliary
        # forwards, including dropout RNG consumption. They do not add a loss.
        aux = row.get('auxiliary', {})
        if aux.get('eligible'):
            anchor = aux['anchor']
            template = cfg['aux_prompt'].replace('START', str(anchor['start'])).replace('END', str(anchor['end']))
            ai = engine.inputs(image_path, row['caption'], template)
            with torch.no_grad():
                for text in [anchor['text'], aux['negative_literal_quote']]:
                    # Complete quote tokens, with empty prefix, exactly as the retained control.
                    engine.branch(ai, text, prefix='')
        if (i + 1) % 8 == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.)
            opt.step()
            opt.zero_grad(set_to_none=True)
    assert len(rows) % 8 == 0
    model.save_pretrained(args.output / 'adapter', safe_serialization=True)
    (args.output / 'training.json').write_text(json.dumps(dict(config=cfg, history=log), indent=2))


if __name__ == '__main__':
    main()
