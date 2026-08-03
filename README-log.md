# ImmuneCLIP Artifact Log

This directory is the canonical artifact workspace for ImmuneCLIP experiments from 2026-08-03 onward.

## Directory Layout

- `baselines/attack_baselines/`: attack baseline source code and wrappers.
- `baselines/defense_baselines/`: defense baseline source code and wrappers.
- `immuneclip/`: ImmuneCLIP core source code.
- `logs/`: experiment logs, grouped by experiment name.
- `outputs/`: summaries, tables, and figures, grouped by experiment name.
- `scripts/`: reproducible command wrappers.
- `configs/`: frozen configs and run manifests.
- `docs/`: experiment notes and reports.

## 2026-08-03 Initialization

Source of truth copied from `/root/workspace/usenix/scripts` on `autodl-48G-2`.
The current ImmuneCLIP implementation is the single-proxy version in `immuneclip/immuneclip_new_train.py`.

Initial copied entry points:

- `immuneclip/immuneclip_new_train.py`: single-proxy ImmuneCLIP training.
- `immuneclip/run_downstream.py`: downstream adaptation and rebound evaluation.
- `immuneclip/stage0_blackbox_invert.py`: scan-then-invert proxy construction.
- `immuneclip/run_gradient_rebound_causality.py`: E2 gradient projection causality experiment.

Existing validated summary copied:

- `outputs/single_proxy_formal_rebound_summary.json`


## 2026-08-03 Dependency Completion

Added shared runtime dependencies under : , , , , and .

## 2026-08-03 Repository Slimming

Large generated baseline data and historical visualization outputs are excluded from git. Baseline source code remains under baselines/. Runtime data/checkpoints are referenced by absolute experiment paths in run logs and configs.

## 2026-08-03 Main Experiment Launch

Experiment checklist source: `docs/experiment_checklist.md`.

| ID | Server | Status | Entry | Key inputs | Logs | Outputs | Notes |
|---|---|---|---|---|---|---|---|
| E1 | autodl-48G-1 | running | `bash scripts/run_E1_align_main_seed42.sh` | ATK-1 verified poisoned ckpt, PAR ckpt, InverTune ckpt, clean RN50; strict no-banana CC3M CSV | `logs/E1_align_main/`, `logs/E1_align_main_rerun1/nohup_seed42.log` | `outputs/E1_align_main/` | Initial wrong No-defense ckpt was stopped; rerun uses week2 verified poisoned checkpoint with step-0 ASR near 0.9. |
| E2 | autodl-48G-2 | running | `bash scripts/run_E2_causality_seed42.sh` | PAR-cleaned ATK-1 ckpt; oracle diagnostic patch; proxy trigger; strict no-banana CC3M CSV | `logs/E2_causality_main_fix1/` | `outputs/E2_causality_main_fix1/` | Fixed `random_match_scale` logging bug; failed pre-fix log preserved under `logs/E2_causality_main/`. |

Planned table fields to fill after completion: step-wise ASR/CA at `0,5,10,20,30,50,100,200,300`, `A_post`, `Rebound-Delta`, normalized AURC, revival step, and available gradient diagnostic records.
