# Implementation provenance

Portable entry points retain the paper interfaces and objectives. Local-directory dependencies were replaced with explicit CLI arguments and versioned manifests. CPU checkpoint replay and prediction-table regression are included in validation. GPU training entry points were packaged and inspected; no new training was launched for this release.

| Release entry point / experiment module | Retained source filename | Original SHA-256 |
|---|---|---|
| `inference/full_label.py` — Full-label scoring | `cab_common.py` | `64a365f05278391b46c09d0390a66b4cd71814da929e6f5938e4c7f0c29a6e79` |
| `inference/readouts.py` — Frozen readout inference | `mms_phase3_common.py` | `61631eac8aac4309dd0c404b2f4b4ae771883899b9b169c8593d41860220d06f` |
| `training/readouts.py` — Frozen readout learning | `mms_phase2_readouts.py` | `248869c1350c551f844ff9282790d18f6d4d3fe77f56ae3dea2e50e8bee3934c` |
| `training/qlora.py` — Classification adaptation | `train_evaluate.py` | `adfe1d7b98bc43d977c890f3c7785e29a5d5374171fba63b2a2a40c4e0271bcd` |
| `training/joint.py` — Joint adaptation | `train_joint.py` | `7b6a3f0d322d1901d9b6818dba377ce1b9b4c08009dbaa20321bf1f7e567a474` |
| `evaluation/metrics.py` — Evaluation reconstruction | `mms_phase3_analyze.py` | `c476584dbee7606e687811833bcdcdbd3d02c50f57b5d1ce8cbc9148d8e7c7b5` |

The hashes identify the retained source versions before portability edits; they are not hashes of the migrated release entry points. Original filenames are retained for version traceability. The modules above were migrated to explicit resource paths. CPU validation covers readout checkpoint replay, metrics, and prediction-table reconstruction. Classification adaptation, joint adaptation, and full VLM inference were checked at source/interface level only; GPU training and inference were not repeated.

The full-label score sums log probabilities over complete labels, including both branches. Joint training leaves this score differentiable. The initialization trainer preserves the original no-gradient auxiliary quote forwards, including their RNG consumption; they contribute no auxiliary classification loss. Exported initialization schedules retain the frozen order.

The readout trainer supports two distinct selection schedules: matched-objective epoch checks and fixed-update architecture-matrix checks. All 80 matrix fits receive 1,280 updates; the small-scale samples follow the original seed-specific nested selection. GPU portability is a source/interface check, not a claim that the original hardware run was repeated.
