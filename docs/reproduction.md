# Reproduction guide

## Core result reconstruction

```bash
python scripts/reproduce.py --data /path/to/evaluation-resources --output outputs/tables
```

Outputs cover matched objectives, simple heads, separate scale/adaptation confirmations, 80 architecture fits (development and exposed evaluation), human-reference metrics, and SugarCrepe categories. Point estimates are calculated from condition-level scores. Frozen labels and archived statistics are regression targets, rather than substituted outputs.

For the grouped uncertainty calculation use `evaluation.metrics.paired_bootstrap`: provide aligned image IDs, effective binary labels, the two score vectors, and their frozen thresholds. Its draws sample whole image groups. `protocols/evaluation.json` and the retained `statistics/*bootstrap*` files specify comparison-specific seeds, repetitions, and multiplicity adjustments. The main pAUC is the trapezoidal ROC integral on `[0, beta] / beta`; it has no chance correction. ROC recall at FAR≤5% and development-threshold recall are separate outputs.

## Model interface

```text
Image + single exact Caption
  -> native Qwen processor/chat template (4096–262144 pixels)
  -> Qwen3-VL-4B NF4 + Classification QLoRA
  -> h: [2560] before the first candidate-label token
     z: scalar, logP(full CONFLICT) - logP(full NO_CONFLICT)
  -> training-set standardizer
  -> Linear h: -Linear(h), or MLP h: -MLP(h)
     Residual h,z: z - MLP(h)
  -> scalar conflict score -> frozen development threshold
```

Both label sequences are teacher-forced in full. One scored input uses both branches; the hidden state comes from the common pre-label prefix, not the answer. The MLP is 2560→32→1 with Tanh. Neither inference nor readouts use a sibling caption, edit position, or source label. Thresholds in `models.json` are the retained human-reference protocol thresholds; individual historical tables retain their own frozen thresholds. Do not exchange thresholds among cohorts merely because checkpoint names match.

## CPU fitting

```bash
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES='' python -m training.readouts --features resources/features \
  --config configs/linear.json --output outputs/linear-new
CUDA_VISIBLE_DEVICES='' python -m training.readouts --features resources/features \
  --config configs/ranking.json --output outputs/ranking-new
CUDA_VISIBLE_DEVICES='' python -m training.scaling --features resources/features \
  --config configs/scaling.json --output outputs/matrix-new
```

Each run uses a new output directory. Four scales are 256/512/1024/1536 image groups, with seed-specific nested subsets at the smaller sizes. The matrix fixes 1,280 optimizer updates, 64 pairs/update, and development selection at step 16 and every 80 steps. Matched objectives use the original epoch-based selection cadence. Normal/conflict conditions stay paired and development IDs stay separate. Standardization is fitted only on the selected training conditions. Configuration files preserve objective, seed, and selection identity; no evaluation score chooses a checkpoint.

## Adapter training entry points

```bash
python -m training.qlora --base-model /path/to/Qwen3-VL-4B-Instruct \
  --image-root /path/to/CLASH-images --config configs/classification.json \
  --manifest resources/protocols/classification_schedule.json --output outputs/adapter-init
python -m training.joint --base-model /path/to/Qwen3-VL-4B-Instruct \
  --image-root /path/to/CLASH-images --assets assets/v1 --config configs/joint.json \
  --manifest resources/protocols/joint_groups.json --output outputs/joint-adaptation
```

The frozen schedule preserves initialization order and no-gradient quote-control forwards. File names in the schedules correspond to the retained CLASH image layout; resolve them to locally obtained images before starting. Joint adaptation updates language-side q/v LoRA and the residual head; NF4 base weights, vision, merger, and the original standardizer remain frozen. Both h and the two-branch z are live differentiable tensors. The configuration covers 1,536 training images, three epochs, 9,216 conditions, and 288 updates. The GPU commands check availability and leave busy resources untouched. Runtime versions are recorded in the requirements and provenance documents. New GPU training was not executed for packaging.

## Diagnostics and validation

```bash
python -m analysis.pair_decision --predictions resources/predictions/readout_scores/matched_objectives.csv \
  --output outputs/pair-decision.csv
python -m analysis.pair_center --predictions resources/predictions/readout_scores/matched_objectives.csv \
  --model classification
python -m analysis.threshold_sensitivity \
  --predictions resources/predictions/readout_scores/architecture_scaling.csv.gz \
  --model residual_hz-1536-2026102813 --output outputs/calibration.csv
CUDA_VISIBLE_DEVICES='' python tests/test_release.py --data resources --assets assets/v1 \
  --output outputs/cpu-validation.json
```

Check model keys in the relevant CSV: architecture matrix names follow the archived keys. Pair-center subtraction uses sibling information and is a diagnostic. Cross-image violations exclude the same image's sibling; ROC metrics compare all valid binary conditions. Calibration resampling varies only normal development examples, keeping evaluation scores unchanged. The tests use `requirements-test.txt` and cover tied-score metrics, strict thresholding, qualified pairs, asset integrity, and replay of all six human-reference models.

## Evaluation identities

The 512-image readout evaluation cohort is exposed in later exploratory analyses. Scale confirmation and adaptation confirmation are two separate, first-scored 512-image cohorts. Human-reference evaluation provides full, model-specific exclusion, and common 245-image exclusion scopes, with random and diagnostic cohorts reported separately. Human INSUFFICIENT cases remain in the ledger and are excluded from both source/human binary comparisons using the same mask. SugarCrepe uses strict native positive-vs-negative caption choice over all 7,511 planned pairs; ties and failures count as incorrect.

## Independent scale-confirmation interval

```bash
python scripts/bootstrap_comparison.py \
  --predictions resources/predictions/readout_scores/scale_confirmation.csv \
  --model-a E1 --model-b E0 --repetitions 10000 --seed 2026102820 \
  --alpha 0.025 --output outputs/scale-bootstrap.json
```

The retained paired pAUC difference is 0.02600098, with adjusted 97.5% interval [0.00333385, 0.04714222]. `docs/scale-interval-validation.json` records the exact CPU reconstruction.
