"""Portable extraction of the original two-branch, full-label decision interface."""
import json
from pathlib import Path
import torch


class Engine:
    def __init__(self, base_model, adapter=None, trainable=False):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
        from peft import PeftModel
        self.processor = AutoProcessor.from_pretrained(base_model, local_files_only=True,
                                                       min_pixels=4096, max_pixels=262144)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model, local_files_only=True, dtype=torch.bfloat16, device_map='cuda:0',
            attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16))
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=trainable)
        if not all(p.device.type == 'cuda' for p in self.model.parameters()):
            raise RuntimeError('The paper interface uses single-GPU inference without CPU offload')
        self.model.eval()
        self.model.config.use_cache = False
        self.capture = {}
        self.hook = None

    def capture_hidden(self):
        def hook(module, args):
            if 'h' not in self.capture:
                self.capture['h'] = args[0][0, self.capture['position']].float()
        self.hook = self.model.get_output_embeddings().register_forward_pre_hook(hook)

    def inputs(self, image_path, caption, template):
        from PIL import Image
        with Image.open(image_path) as image:
            rgb = image.convert('RGB')
        prompt = template.replace('CAPTION_JSON', json.dumps(caption, ensure_ascii=False))
        return self.processor.apply_chat_template([{'role': 'user', 'content': [
            {'type': 'image', 'image': rgb}, {'type': 'text', 'text': prompt}]}],
            tokenize=True, add_generation_prompt=True, return_tensors='pt',
            return_dict=True).to('cuda:0')

    def branch(self, inputs, label, prefix='{"decision":"'):
        tok = self.processor.tokenizer
        pre = tok(prefix, add_special_tokens=False)['input_ids']
        target = tok(label, add_special_tokens=False)['input_ids']
        n = inputs['input_ids'].shape[1]
        out = dict(inputs)
        ids = pre + target
        out['input_ids'] = torch.cat([inputs['input_ids'], torch.tensor([ids], device='cuda')], 1)
        out['attention_mask'] = torch.ones_like(out['input_ids'])
        if 'mm_token_type_ids' in out:
            out['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'], torch.zeros(
                (1, len(ids)), device='cuda', dtype=inputs['mm_token_type_ids'].dtype)], 1)
        pos = torch.arange(n + len(pre) - 1, n + len(ids) - 1, device='cuda')
        logits = self.model(**out, use_cache=False).logits[0, pos].float()
        return -torch.nn.functional.cross_entropy(logits, torch.tensor(target, device='cuda'), reduction='sum')

    def live(self, inputs):
        prefix = self.processor.tokenizer('{"decision":"', add_special_tokens=False)['input_ids']
        self.capture.clear()
        self.capture['position'] = inputs['input_ids'].shape[1] + len(prefix) - 1
        z = self.branch(inputs, 'CONFLICT') - self.branch(inputs, 'NO_CONFLICT')
        return self.capture['h'], z

    def close(self):
        if self.hook:
            self.hook.remove()
