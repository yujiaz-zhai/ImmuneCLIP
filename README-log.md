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

E2 all-param causality completed on `autodl-48G-2` as a single-seed run (`seed=42`).

| Intervention | A0 | A_post | Delta R | AURC | CA_T | T_0.5 | Output |
|---|---:|---:|---:|---:|---:|---:|---|
| Normal benign update | 0.075 | 0.405 | 0.330 | 0.2855 | 0.517 | — | `outputs/E2_causality_allparam_fix2/rn50_align_par_normal_allparam_s42/` |
| Reactivation-projected | 0.075 | 0.133 | 0.058 | 0.0037 | 0.510 | — | `outputs/E2_causality_allparam_fix2/rn50_align_par_project_oracle_allparam_s42/` |
| Matched-component random | 0.075 | 0.267 | 0.192 | 0.1754 | 0.512 | — | `outputs/E2_causality_allparam_fix2/rn50_align_par_random_matched_allparam_s42/` |
| Shuffled-proxy direction | 0.075 | 0.275 | 0.200 | 0.1689 | 0.511 | — | `outputs/E2_causality_allparam_fix2/rn50_align_par_proxy_shuffled_allparam_s42/` |

Interpretation: removing the oracle reactivation component suppresses rebound much more strongly than removing a matched-size random component or a shuffled proxy direction, while CA@1 remains comparable. The current result is sufficient for a single-seed pilot; the checklist's final camera-ready standard still requires paired 3-seed confidence bands.

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

## 2026-08-03 BadNet Attack Candidate Correction

The initial ATK-3 BadNet-random candidate used `1500/500000` poisoned samples, inherited from the optimized BadCLIP patch setting. It was stopped after epoch 1 because the logged ASR was only `0.0013`, so it is recorded as a failed attack candidate and will not be used as the正文 BadNet row.

Replacement launched on `autodl-48G-2`:

| Candidate | Trigger | Poison count | Epochs | Warmup | Status | Logs |
|---|---|---:|---:|---:|---|---|
| BadNet-random-5pct | random 16x16 patch, random location, target banana | 25000 / 500000 | 5 | 1000 steps | running | `logs/E1_badnet_attack_strong/` |

The BadNet defense queue now waits for `logs/E1_badnet_attack_strong/poison_logs/rn50_badnet_random_p5pct_ep5_s42/checkpoints/epoch_5.pt`.

### 2026-08-03 12:16 CST - E1/E2 completion queue update
- User requirement clarified: fill every blank in the E1 matrix, including previously unimplemented BadNet baselines, and fill E2 shuffled-proxy control.
- E1 BadNet weak candidate rn50_badnet_random_poison_ep10_s42 stopped as invalid candidate after epoch_1 ASR=0.0013; replacement rn50_badnet_random_p5pct_ep5_s42 is running from a 5% poisoned random-patch training set.
- BadNet defense queue waits for epoch_5.pt and will run No-defense, PAR, CleanCLIP, InverTune, and PAR proj/partial rebound evaluations with random patch metadata.
- E2 shuffled-proxy intervention implemented as proxy gradient tensor shuffling and committed in b9c33e2; queued on server-1 after current E1 jobs.

## 2026-08-03 E2/E3 Checklist Tables

Scope: single-seed pilot results with `seed=42`. `A_post` is the maximum ASR observed over the downstream trajectory. `Delta R` is `max_t(ASR_t - ASR_0)`. `AURC` is the ASR trajectory area recorded by `run_downstream.py`. Rows marked `not run` are experiment-list cells that do not yet have real logged measurements and must not be cited as completed.

### E2: Gradient Projection Causality

| Intervention | A0 | A_post | Delta R | AURC | CA_T | T_0.5 | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| Normal benign update | 0.075 | 0.405 | 0.330 | 0.2855 | 0.517 | — | `outputs/E2_causality_allparam_fix2/rn50_align_par_normal_allparam_s42/rn50_align_par_normal_allparam_s42_summary.json` |
| Reactivation-projected | 0.075 | 0.133 | 0.058 | 0.0037 | 0.510 | — | `outputs/E2_causality_allparam_fix2/rn50_align_par_project_oracle_allparam_s42/rn50_align_par_project_oracle_allparam_s42_summary.json` |
| Matched-component random | 0.075 | 0.267 | 0.192 | 0.1754 | 0.512 | — | `outputs/E2_causality_allparam_fix2/rn50_align_par_random_matched_allparam_s42/rn50_align_par_random_matched_allparam_s42_summary.json` |
| Shuffled-proxy direction | 0.075 | 0.275 | 0.200 | 0.1689 | 0.511 | — | `outputs/E2_causality_allparam_fix2/rn50_align_par_proxy_shuffled_allparam_s42/rn50_align_par_proxy_shuffled_allparam_s42_summary.json` |

E2 interpretation: removing the oracle reactivation component suppresses rebound much more strongly than removing a matched-size random component or shuffled proxy direction, while final CA remains close.

### E3: ImmuneCLIP Main Table

Current E3 status: the completed `Align + PAR + ImmuneCLIP` full-FT row is logged but **not yet达标**. It lowers immediate ASR but still rebounds to `A_post=0.408`; this row is kept as an honest failed/weak pilot, not as the final method result.

| Attack | Purifier | Method | Adapt | CA0 | A0 | A_post | Delta R | AURC | CA_T | rho_SP | GPU-h | Source |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| Align | PAR | purifier only | full | 0.520 | 0.075 | 0.559 | 0.484 | 0.4922 | 0.497 | not logged | — | E1 Align/PAR full baseline reused as E3 purifier-only control |
| Align | PAR | + compute-matched FT | full | 0.491 | 0.146 | 0.554 | 0.408 | 0.5006 | 0.500 | 0.0253 final `dir_raw_max` | 0.30 | `outputs/E3_align_par_main_seed42/runs/rn50_align_par_compute_matched_cleanft_s42/`, `outputs/E3_align_par_main_seed42/downstream/rn50_align_par_compute_matched_cleanft_s42_rebound_full/` |
| Align | PAR | + ImmuneCLIP | full | 0.464 | 0.014 | 0.408 | 0.394 | 0.3640 | 0.502 | 0.5341 final `dir_raw_max` | 0.11 | `outputs/E3_align_par_main_seed42/runs/rn50_align_par_immuneclip_checkpoint_rho_s42/`, `outputs/E3_align_par_main_seed42/downstream/rn50_align_par_immuneclip_checkpoint_rho_s42_rebound_full/` |
| Align | InverTune | purifier only | full | 0.569 | 0.000 | 0.828 | 0.828 | 0.7462 | 0.556 | not logged | — | E1 Align/InverTune full baseline reused as purifier-only control |
| Align | InverTune | + compute-matched FT | full | not run | not run | not run | not run | not run | not run | not run | not run | pending |
| Align | InverTune | + ImmuneCLIP | full | not run | not run | not run | not run | not run | not run | not run | not run | pending |
| BadCLIP | PAR | purifier only | full | 0.535 | 0.069 | 0.469 | 0.400 | 0.4021 | 0.519 | not logged | — | E1 BadCLIP/PAR full baseline reused as purifier-only control |
| BadCLIP | PAR | + ImmuneCLIP | full | not run | not run | not run | not run | not run | not run | not run | not run | pending |
| BadCLIP | InverTune | purifier only | full | 0.572 | 0.000 | 0.645 | 0.645 | 0.5587 | 0.568 | not logged | — | E1 BadCLIP/InverTune full baseline reused as purifier-only control |
| BadCLIP | InverTune | + ImmuneCLIP | full | not run | not run | not run | not run | not run | not run | not run | not run | pending |
| Align | PAR | purifier only | proj | 0.520 | 0.075 | 0.553 | 0.478 | 0.5006 | 0.506 | not logged | — | E1 Align/PAR projection baseline reused as purifier-only control |
| Align | PAR | + ImmuneCLIP | proj | not run | not run | not run | not run | not run | not run | not run | not run | pending |
| Align | InverTune | purifier only | proj | 0.569 | 0.000 | 0.816 | 0.816 | 0.7383 | 0.562 | not logged | — | E1 Align/InverTune projection baseline reused as purifier-only control |
| Align | InverTune | + ImmuneCLIP | proj | not run | not run | not run | not run | not run | not run | not run | not run | pending |

E3 issue note: the current `checkpoint_rho` ImmuneCLIP pilot is not final. It reduces immediate ASR from `0.075` to `0.014`, but full benign FT rebounds to `0.408`. The next repair target is to rerun a working `traj_global` single-proxy variant inside artifacts and then re-evaluate 300-step rebound.
