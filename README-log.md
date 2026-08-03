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

## 2026-08-03 Coverage Correction And Missing Baseline Launch

The previous E1 status covered only the ATK-1 Align-surrogate main line. It did **not** complete every attack baseline in the E1 table.

Current E1 table coverage:

| Attack | Purifier | Adapt | Status | Log / source |
|---|---|---|---|---|
| Align-surrogate | None | full | complete | `outputs/E1_align_main/` |
| Align-surrogate | PAR | full | complete | `outputs/E1_align_main/` |
| Align-surrogate | InverTune | full | complete | `outputs/E1_align_main/` |
| C1 clean control | None | full | complete | `outputs/E1_align_main/` |
| BadCLIP | None | full | running/partially complete | `logs/E1_badclip_main/`, `outputs/E1_badclip_main/` |
| BadCLIP | PAR | full | running/partially complete | `logs/E1_badclip_main/`, `outputs/E1_badclip_main/` |
| BadCLIP | CleanCLIP | full | running | `logs/E1_badclip_main/`, `outputs/E1_badclip_main/` |
| BadCLIP | InverTune | full | missing defended checkpoint; must rerun InverTune defense | `baselines/defense_baselines/InverTune/results/badclip_banana_paper/evaluation/defended_metrics.json` points to a non-existing checkpoint path |
| BadNet | None/CleanCLIP/PAR/InverTune | full | missing; BadNet attack implementation launched first | `scripts/run_E1_badnet_attack_seed42.sh`, `logs/E1_badnet_attack/` |
| Align-surrogate / BadCLIP | PAR/InverTune | proj/partial | script prepared, not yet launched | `scripts/run_E1_projection_existing_seed42.sh` |

Code/interface updates:

- `immuneclip/clip_eval.py` now accepts explicit `target_label`, `patch_type`, `patch_location`, `patch_size`, and `patch_name` arguments while preserving the BadCLIP default.
- `immuneclip/run_downstream.py` exposes the same attack-trigger fields through CLI and writes them into trajectory summaries.
- This is required for BadNet/structured-trigger rows; otherwise all non-BadCLIP attacks would be evaluated with the wrong trigger.

New scripts:

- `scripts/run_E1_badclip_main_seed42.sh`: unified strict no-banana rebound trajectories for BadCLIP None/PAR/CleanCLIP.
- `scripts/run_E1_badnet_attack_seed42.sh`: generates `backdoor_banana_random_random_16_500000_1500.csv` and trains RN50 BadNet-random poisoned checkpoint.
- `scripts/run_E1_projection_existing_seed42.sh`: runs existing Align/BadCLIP purifier checkpoints under the current partial/proj adaptation implementation.

E2 all-param causality completed on `autodl-48G-2`:

| Intervention | A0 | A_post | Delta R | CA_T | Output |
|---|---:|---:|---:|---:|---|
| Normal benign update | 0.075 | 0.405 | 0.330 | 0.517 | `outputs/E2_causality_allparam_fix2/rn50_align_par_normal_allparam_s42/` |
| Reactivation-projected | 0.075 | 0.133 | 0.058 | 0.510 | `outputs/E2_causality_allparam_fix2/rn50_align_par_project_oracle_allparam_s42/` |
| Matched-component random | 0.075 | 0.267 | 0.192 | 0.512 | `outputs/E2_causality_allparam_fix2/rn50_align_par_random_matched_allparam_s42/` |

Interpretation: removing the oracle reactivation component suppresses rebound much more strongly than removing a matched-size random component, while CA@1 remains comparable.

## 2026-08-03 E1 Coverage Expansion Status Update

Goal: fill every non-optional empty cell in the E1 table with real logged measurements. Previous E1 status covered the Align-surrogate main line and did not complete all attack baselines.

Current allocation:

| Server | Queue | Status | Logs |
|---|---|---|---|
| autodl-48G-1 | E1 existing projection/partial rows: Align+PAR, Align+InverTune, BadCLIP+PAR | running from `scripts/run_E1_projection_existing_seed42.sh` | `logs/E1_projection_existing/`, `outputs/E1_projection_existing/` |
| autodl-48G-1 | E1 BadCLIP+InverTune full rebound | queued after projection PID 10778 | `logs/E1_badclip_invertune/`, `outputs/E1_badclip_invertune/` |
| autodl-48G-2 | E1 BadNet random attack training | running after fixing incomplete CLIP cache; uses ATK-3 random trigger | `logs/E1_badnet_attack/` |
| autodl-48G-2 | E1 BadNet defenses and rebound: No-defense, PAR, CleanCLIP, InverTune, PAR proj/partial | queued and waiting for `epoch_10.pt` | `logs/E1_badnet_defenses/`, `outputs/E1_badnet_defenses/` |

Completed new E1 rows since the coverage correction:

| Attack | Purifier | Adapt | A0 | Apost | DeltaR | AURC | CAT | Source |
|---|---|---|---:|---:|---:|---:|---:|---|
| BadCLIP | None | full | 0.824 | 0.919 | 0.095 | 0.904025 | 0.567 | `outputs/E1_badclip_main/results/traj_rn50_badclip_nodef_full_s42_contrastive_full_s42_summary.json` |
| BadCLIP | PAR | full | 0.069 | 0.469 | 0.400 | 0.402142 | 0.519 | `outputs/E1_badclip_main/results/traj_rn50_badclip_par_full_s42_contrastive_full_s42_summary.json` |
| BadCLIP | CleanCLIP | full | 0.025 | 0.056 | 0.031 | 0.045575 | 0.438 | `outputs/E1_badclip_main/results/traj_rn50_badclip_cleanclip_full_s42_contrastive_full_s42_summary.json` |

Implementation notes:

- `immuneclip/clip_eval.py` and `immuneclip/run_downstream.py` now accept explicit trigger metadata so BadNet/structured trigger rows are not accidentally evaluated with the BadCLIP image patch.
- Current `--ft lora` in `run_downstream.py` is recorded as `lora_partial`; E1 `proj` rows should be described as projection/partial unless a strict projection-only mode is added and rerun.
- BadNet+InverTune will use InverTune for inversion/tuning only; true BadNet ASR and rebound are evaluated by the unified downstream script with `patch_type=random`.
