# From Pairwise Signals to Screening Decisions

**Supervised Readouts and Proxy Misalignment in Image–Text Conflict Detection**

[Evaluation resources](https://huggingface.co/datasets/shbiao/ImageText-Conflict-Screening-Evaluation) · [Reproduction guide](docs/reproduction.md) · [Implementation provenance](docs/implementation-provenance.md)

## Research Overview

How can a vision–language model's response to a conflicting caption become an effective low-false-alarm screening decision? We study supervised decision interfaces over Qwen3-VL-4B Classification QLoRA representations, connecting within-image caption ordering, global score ranking, and frozen-threshold screening.

The repository includes linear, nonlinear, and residual readouts; matched BCE, ranking, and pair-anchor objectives; an 80-fit architecture–supervision study; joint LoRA/readout adaptation; and evaluation against adjudicated human judgments and SugarCrepe caption comparisons.

## Key Findings

- **Supervised readouts improve low-FAR discrimination.** On the readout evaluation cohort, integral-normalized pAUC at FAR 0–10% increases from 0.6693 for the original likelihood contrast to 0.7585 for residual BCE and 0.7671 for a linear readout. The original adapter and added readouts have different supervision budgets; the controlled comparisons separate architecture, objective, and supervision scale.
- **Lightweight linear probes are competitive with nonlinear alternatives.** Four architectures, four supervision scales, and five seeds form an 80-fit controlled study.
- **Stronger proxies need not yield stronger screening.** Pair ranking, pair-center regression, and source-edit localization measure different properties from global low-FAR discrimination.
- **Human judgments and external transfer distinguish decision interfaces.** The release includes 1,050 adjudicated conditions from 525 images and all 7,511 SugarCrepe comparisons. Linear readout improves native SugarCrepe choice accuracy from 95.06% to 96.14%; the residual alternatives do not show the same transfer gain.

The CLASH-derived binary screening protocol and SugarCrepe's native caption-choice protocol are documented separately in [the evaluation guide](docs/reproduction.md). CLASH's official question-driven QA task is a different interface.

## Quick Start

### Recompute the paper tables on CPU

```bash
git clone https://github.com/SHbiao/ImageText-Conflict-Screening.git
cd ImageText-Conflict-Screening
python -m pip install -r requirements-evaluation.txt huggingface_hub
hf download shbiao/ImageText-Conflict-Screening-Evaluation \
  --repo-type dataset --local-dir resources
python scripts/reproduce.py --data resources --output outputs/paper-tables
```

The result reconstruction needs NumPy, released predictions, and labels; it runs without model weights or GPU inference. The evaluation repository is public and can be downloaded without signing in.

### Score one image–caption input

Download the versioned adapter and readout assets from the public [GitHub v0.1.0 release](https://github.com/SHbiao/ImageText-Conflict-Screening/releases/tag/v0.1.0), or use the [direct model-assets-v1.zip download](https://github.com/SHbiao/ImageText-Conflict-Screening/releases/download/v0.1.0/model-assets-v1.zip). Obtain the Qwen base model through its official distribution and retain its native processor.

```bash
python -m pip install -r requirements-gpu.txt
gh release download v0.1.0 --repo SHbiao/ImageText-Conflict-Screening \
  --pattern 'model-assets-v1.zip' --dir assets-download
python -m zipfile -e assets-download/model-assets-v1.zip assets
python -m inference.score --base-model /path/to/Qwen3-VL-4B-Instruct \
  --assets assets/v1 --model Linear-1024 --image /path/to/image.jpg \
  --caption 'A bus is parked beside the road.' --output outputs/example.jsonl
```

`--input-csv examples/single-input.csv` supports resumable batches. Scores increase toward conflict; decisions use `score > frozen_threshold`. GPU admission checks defer execution when another compute task is active. See the [tensor interface](docs/reproduction.md#model-interface) before using thresholds with a different population.

## Experiments

| Component | Purpose | Entry point |
|---|---|---|
| Classification QLoRA | Learn the retained decision representation | `training/qlora.py` |
| Linear / MLP / Residual | Controlled supervised readouts | `training/readouts.py` |
| Ranking / Anchor | Matched proxy-objective comparisons | `configs/ranking.json`, `configs/anchor.json` |
| Architecture × supervision | Fixed 1,280-update, 80-fit study | `training/scaling.py` |
| Joint LoRA/readout | Differentiable full-label adaptation | `training/joint.py` |
| Human / SugarCrepe | Semantic reassessment and external choice | `evaluation/` |
| Pair / global / threshold | Decision-stage diagnostics | `analysis/` |

## Data and Models

| Location | Contents |
|---|---|
| [GitHub repository](https://github.com/SHbiao/ImageText-Conflict-Screening) | Training, inference, evaluation and statistical reconstruction scripts; configurations and reproduction instructions. |
| [GitHub Releases: v0.1.0](https://github.com/SHbiao/ImageText-Conflict-Screening/releases/tag/v0.1.0) / [model-assets-v1.zip](https://github.com/SHbiao/ImageText-Conflict-Screening/releases/download/v0.1.0/model-assets-v1.zip) | Approximately 23 MB: frozen Qwen classification LoRA, joint-adaptation LoRA, supervised readout weights, and readout standardizers. Neither Qwen base-model weights nor original COCO images are included. |
| [Hugging Face Dataset](https://huggingface.co/datasets/shbiao/ImageText-Conflict-Screening-Evaluation) | Anonymized human annotations and adjudication labels, split and model-contact manifests, frozen predictions, cached features, statistics, and evaluation reconstruction resources. No model checkpoints are hosted in this Dataset. |

The evaluation files and model bundle currently cover the primary Qwen experiments. Images are retrieved from their original COCO sources; the Qwen base model is obtained from its official distribution. Third-party models and data retain their original terms.

The unchanged model bundle is 23,173,918 bytes. SHA-256:

```text
7d49ca8741d93180b62c19532c24af759d3157eabbde369374f336fd2e7d2811
```

## Validation

`scripts/reproduce.py` reconstructs the principal prediction tables, all 72 source/human metric rows, and all 32 SugarCrepe category/model rows. CPU checkpoint replay covers six models × 1,050 conditions, with maximum score error below `5e-7` and identical threshold predictions. Validation records are in `docs/cpu-validation.json` and `docs/table-validation.json`. GPU training entry points were ported from the retained implementations and inspected; GPU training and inference were not rerun for this release.

## Citation

**From Pairwise Signals to Screening Decisions: Supervised Readouts and Proxy Misalignment in Image–Text Conflict Detection.** Unpublished manuscript prepared for Multimedia Systems. See `CITATION.cff` for software citation metadata.

Repository maintainer: shbiao <shbiao@stu.nepu.edu.cn> ([SHbiao](https://github.com/SHbiao)).

## License

Original repository code is MIT-licensed. Data, images, and Qwen weights retain their respective terms; the resource card records component-specific licensing and provenance.
