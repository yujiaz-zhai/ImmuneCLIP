# 2026-08-03 E1/E2/E3 Experiment Log

Canonical workspace: `/root/autodl-tmp/experiments/artifacts` on `autodl-48G-2`.

This document consolidates experiments run on 2026-08-03 CST. Lightweight logs and result summaries from `autodl-48G-1` were synchronized into the canonical `autodl-48G-2` artifact workspace. Large checkpoint files (`*.pt`, `*.pth`) are intentionally not committed; their paths are recorded in run logs and summaries. The log inventory below excludes checkpoint blobs.

## Status Snapshot

- `autodl-48G-1`: no E1/E2/E3 training or downstream process was running at the final check.
- `autodl-48G-2`: no E1/E2/E3 training or downstream process was running at the final check.
- Existing pushed commits before this consolidation include `40dbe50` for E3 traj-global rescue and `7fea095` for E1 BadNet-RS-Fixed results.
- This consolidation adds the missing lightweight E1/E2/E3 logs/results from server 1 and this daily log document.

## E1 Rebound Baselines

Metrics: `A0=ASR at downstream step 0`, `A_post=max ASR over trajectory`, `Delta R=max_t(ASR_t-ASR_0)`, `CA_T=final clean accuracy`. All downstream runs use the strict no-banana CC3M CSV unless noted otherwise.

| Row | Status | A0 | A_post | Delta R | AURC | CA0 | CA_T | Summary |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Align / No defense / full | complete | 0.8920 | 0.9630 | 0.0710 | 0.9527 | 0.5750 | 0.5590 | `outputs/E1_align_main/results/traj_rn50_align_nodef_full_s42_contrastive_full_s42_summary.json` |
| Align / PAR / full | complete | 0.0750 | 0.5590 | 0.4840 | 0.4922 | 0.5200 | 0.4970 | `outputs/E1_align_main/results/traj_rn50_align_par_full_s42_contrastive_full_s42_summary.json` |
| Align / InverTune / full | complete | 0.0000 | 0.8280 | 0.8280 | 0.7462 | 0.5690 | 0.5560 | `outputs/E1_align_main/results/traj_rn50_align_invertune_full_s42_contrastive_full_s42_summary.json` |
| Clean RN50 control / No defense / full | complete | 0.0000 | 0.0010 | 0.0010 | 0.0010 | 0.5890 | 0.5560 | `outputs/E1_align_main/results/traj_rn50_clean_control_full_s42_contrastive_full_s42_summary.json` |
| BadCLIP / No defense / full | complete | 0.8240 | 0.9190 | 0.0950 | 0.9040 | 0.5840 | 0.5670 | `outputs/E1_badclip_main/results/traj_rn50_badclip_nodef_full_s42_contrastive_full_s42_summary.json` |
| BadCLIP / PAR / full | complete | 0.0690 | 0.4690 | 0.4000 | 0.4021 | 0.5350 | 0.5190 | `outputs/E1_badclip_main/results/traj_rn50_badclip_par_full_s42_contrastive_full_s42_summary.json` |
| BadCLIP / CleanCLIP / full | complete | 0.0250 | 0.0560 | 0.0310 | 0.0456 | 0.4760 | 0.4380 | `outputs/E1_badclip_main/results/traj_rn50_badclip_cleanclip_full_s42_contrastive_full_s42_summary.json` |
| BadCLIP / InverTune / full | complete | 0.0000 | 0.6450 | 0.6450 | 0.5587 | 0.5720 | 0.5680 | `outputs/E1_badclip_invertune/results/traj_rn50_badclip_invertune_full_s42_contrastive_full_s42_summary.json` |
| Align / PAR / projection-partial | complete | 0.0750 | 0.5530 | 0.4780 | 0.5006 | 0.5200 | 0.5060 | `outputs/E1_projection_existing/results/traj_rn50_align_par_proj_s42_contrastive_lora_s42_summary.json` |
| Align / InverTune / projection-partial | complete | 0.0000 | 0.8160 | 0.8160 | 0.7383 | 0.5690 | 0.5620 | `outputs/E1_projection_existing/results/traj_rn50_align_invertune_proj_s42_contrastive_lora_s42_summary.json` |
| BadCLIP / PAR / projection-partial | complete | 0.0690 | 0.4770 | 0.4080 | 0.4098 | 0.5350 | 0.5050 | `outputs/E1_projection_existing/results/traj_rn50_badclip_par_proj_s42_contrastive_lora_s42_summary.json` |
| Clean RN50 / PAR / full | complete | 0.0010 | 0.0010 | 0.0000 | 0.0001 | 0.5430 | 0.4970 | `outputs/E1_clean_purified_controls/results/traj_rn50_clean_par_full_s42_contrastive_full_s42_summary.json` |
| Clean RN50 / InverTune-noop / full | complete | 0.0000 | 0.0010 | 0.0010 | 0.0010 | 0.5890 | 0.5540 | `outputs/E1_clean_purified_controls/results/traj_rn50_clean_invertune_noop_full_s42_contrastive_full_s42_summary.json` |
| BadNet weak candidate / No defense / full | complete | 0.0140 | 0.0320 | 0.0180 | 0.0277 | 0.5860 | 0.5650 | `outputs/E1_badnet_defenses/results/traj_rn50_badnet_nodef_full_s42_contrastive_full_s42_summary.json` |
| BadNet-RS-Fixed / No defense / full | complete | 0.0840 | 0.4090 | 0.3250 | 0.3718 | 0.5620 | 0.5540 | `outputs/E1_badnet_rs_fixed_defenses/results/traj_rn50_badnet_rs_fixed_nodef_full_s42_contrastive_full_s42_summary.json` |
| BadNet-RS-Fixed / PAR / full | complete | 0.0500 | 0.3920 | 0.3420 | 0.3652 | 0.5240 | 0.4970 | `outputs/E1_badnet_rs_fixed_defenses/results/traj_rn50_badnet_rs_fixed_par_full_s42_contrastive_full_s42_summary.json` |
| BadNet-RS-Fixed / CleanCLIP / full | complete | 0.1430 | 0.1950 | 0.0520 | 0.1839 | 0.5510 | 0.5320 | `outputs/E1_badnet_rs_fixed_defenses/results/traj_rn50_badnet_rs_fixed_cleanclip_full_s42_contrastive_full_s42_summary.json` |
| BadNet-RS-Fixed / PAR / projection-partial | complete | 0.0500 | 0.3800 | 0.3300 | 0.3611 | 0.5240 | 0.5090 | `outputs/E1_badnet_rs_fixed_defenses/results/traj_rn50_badnet_rs_fixed_par_proj_s42_contrastive_lora_s42_summary.json` |
| BadNet-RS-Fixed / InverTune / full | failed baseline | - | - | - | - | - | - | `logs/E1_badnet_rs_fixed_defenses/invertune_trigger_inversion.log`; best inverted ASR 0.39% < 70%, so no defended checkpoint was produced. |

E1 notes:

- Align and BadCLIP rows are the main persistence baselines. Clean RN50 rows are sanity controls showing no banana rebound under the same benign downstream pipeline.
- The early BadNet random candidates are weak/failed candidates and should not be cited as a strong attack baseline. BadNet-RS-Fixed is usable as a structured-trigger sanity baseline but is only medium-strength: No-defense `A_post=0.409`.
- Current `ft=lora` rows are recorded as projection/partial adaptation in the artifact tables.

## E2 Gradient Projection Causality

| Intervention | Status | A0 | A_post | Delta R | AURC | CA0 | CA_T | Summary |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Normal benign update | complete | 0.0750 | 0.4050 | 0.3300 | 0.2855 | 0.5200 | 0.5170 | `outputs/E2_causality_allparam_fix2/rn50_align_par_normal_allparam_s42/rn50_align_par_normal_allparam_s42_summary.json` |
| Reactivation-projected | complete | 0.0750 | 0.1330 | 0.0580 | 0.0037 | 0.5200 | 0.5100 | `outputs/E2_causality_allparam_fix2/rn50_align_par_project_oracle_allparam_s42/rn50_align_par_project_oracle_allparam_s42_summary.json` |
| Matched-component random | complete | 0.0750 | 0.2670 | 0.1920 | 0.1754 | 0.5200 | 0.5120 | `outputs/E2_causality_allparam_fix2/rn50_align_par_random_matched_allparam_s42/rn50_align_par_random_matched_allparam_s42_summary.json` |
| Shuffled-proxy direction | complete | 0.0750 | 0.2750 | 0.2000 | - | 0.5200 | 0.5110 | `outputs/E2_causality_allparam_fix2/rn50_align_par_proxy_shuffled_allparam_s42/rn50_align_par_proxy_shuffled_allparam_s42_summary.json` |

E2 interpretation: oracle reactivation projection suppresses rebound far more than matched random or shuffled-proxy controls while preserving similar final CA, supporting a causal role for the reactivation gradient component. This remains a single-seed pilot, not a final confidence-band result.

## E3 ImmuneCLIP Method Runs

| Row | Status | A0 | A_post | Delta R | AURC | CA0 | CA_T | Summary |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Align+PAR / compute-matched clean FT / full | complete | 0.1460 | 0.5540 | 0.4080 | 0.5006 | 0.4910 | 0.5000 | `outputs/E3_align_par_main_seed42/downstream/rn50_align_par_compute_matched_cleanft_s42_rebound_full/results/traj_rn50_align_par_compute_matched_cleanft_s42_rebound_full_contrastive_full_s42_summary.json` |
| Align+PAR / ImmuneCLIP checkpoint-rho pilot / full | complete | 0.0140 | 0.4080 | 0.3940 | 0.3640 | 0.4640 | 0.5020 | `outputs/E3_align_par_main_seed42/downstream/rn50_align_par_immuneclip_checkpoint_rho_s42_rebound_full/results/traj_rn50_align_par_immuneclip_checkpoint_rho_s42_rebound_full_contrastive_full_s42_summary.json` |
| Align+PAR / ImmuneCLIP traj-global rescue / full | complete | 0.0000 | 0.1310 | 0.1310 | 0.1060 | 0.4390 | 0.4810 | `outputs/E3_align_par_rescue_seed42/downstream/rn50_align_par_immuneclip_traj_global_working_s42_rebound_full/results/traj_rn50_align_par_immuneclip_traj_global_working_s42_rebound_full_contrastive_full_s42_summary.json` |

E3 notes:

- `checkpoint-rho` is an honest weak/failed pilot and should not be cited as the final method result.
- `traj-global rescue` is the current working single-proxy method candidate: immediate ASR 0.000, max downstream ASR 0.131, final ASR 0.113, final CA 0.481.
- This working row should be described as the single-proxy trajectory approximation, not as full strict reachable-checkpoint ImmuneCLIP.

## Key Entry Points

| Experiment | Entry | Logs | Outputs / checkpoint reference |
|---|---|---|---|
| E1 Align main | `scripts/run_E1_align_main_seed42.sh` | `logs/E1_align_main/` | `outputs/E1_align_main/` |
| E1 BadCLIP main | `scripts/run_E1_badclip_main_seed42.sh` | `logs/E1_badclip_main/` | `outputs/E1_badclip_main/` |
| E1 projection existing | `scripts/run_E1_projection_existing_seed42.sh` | `logs/E1_projection_existing/` | `outputs/E1_projection_existing/` |
| E1 BadCLIP InverTune | `queued wrapper; see master log` | `logs/E1_badclip_invertune/` | `outputs/E1_badclip_invertune/` |
| E1 clean purified controls | `queued wrapper; see master log` | `logs/E1_clean_purified_controls/` | `outputs/E1_clean_purified_controls/` |
| E1 BadNet-RS-Fixed attack | `scripts/run_E1_badnet_rs_fixed_attack_seed42.sh` | `logs/E1_badnet_rs_fixed_attack/` | `logs/E1_badnet_rs_fixed_attack/poison_logs/.../checkpoints/epoch_3.pt` |
| E1 BadNet-RS-Fixed defenses | `scripts/run_E1_badnet_rs_fixed_defenses_seed42.sh` | `logs/E1_badnet_rs_fixed_defenses/` | `outputs/E1_badnet_rs_fixed_defenses/` |
| E2 causality | `scripts/run_E2_causality_seed42.sh / allparam fix2 wrappers` | `logs/E2_causality_allparam_fix2/` | `outputs/E2_causality_allparam_fix2/` |
| E3 main pilots | `scripts/run_E3_align_par_main_seed42.sh` | `logs/E3_align_par_main_seed42/` | `outputs/E3_align_par_main_seed42/` |
| E3 traj-global rescue | `manual rescue wrapper; see master/nohup logs` | `logs/E3_align_par_rescue_seed42/` | `outputs/E3_align_par_rescue_seed42/` |

## Failure / Non-Citable Items

- `outputs/E1_badnet_defenses/results/traj_rn50_badnet_nodef_full_s42...`: weak BadNet random candidate, `A_post=0.032`; retained only for audit.
- `logs/E1_badnet_rs_fixed_defenses/invertune_trigger_inversion.log`: InverTune failed to invert BadNet-RS-Fixed, best inverted ASR 0.39%; no defended checkpoint exists.
- `E3 checkpoint-rho`: logged as weak pilot, not final method evidence.
- Multi-seed confidence bands are not completed today; all tables here are single-seed `seed=42` pilot/benchmark rows.

## Summary File Inventory

- `outputs/E1_align_main/results/traj_rn50_align_invertune_full_s42_contrastive_full_s42_summary.json`
- `outputs/E1_align_main/results/traj_rn50_align_nodef_full_s42_contrastive_full_s42_summary.json`
- `outputs/E1_align_main/results/traj_rn50_align_par_full_s42_contrastive_full_s42_summary.json`
- `outputs/E1_align_main/results/traj_rn50_clean_control_full_s42_contrastive_full_s42_summary.json`
- `outputs/E1_badclip_invertune/results/traj_rn50_badclip_invertune_full_s42_contrastive_full_s42_summary.json`
- `outputs/E1_badclip_main/results/traj_rn50_badclip_cleanclip_full_s42_contrastive_full_s42_summary.json`
- `outputs/E1_badclip_main/results/traj_rn50_badclip_nodef_full_s42_contrastive_full_s42_summary.json`
- `outputs/E1_badclip_main/results/traj_rn50_badclip_par_full_s42_contrastive_full_s42_summary.json`
- `outputs/E1_badnet_defenses/results/traj_rn50_badnet_nodef_full_s42_contrastive_full_s42_summary.json`
- `outputs/E1_badnet_rs_fixed_defenses/results/traj_rn50_badnet_rs_fixed_cleanclip_full_s42_contrastive_full_s42_summary.json`
- `outputs/E1_badnet_rs_fixed_defenses/results/traj_rn50_badnet_rs_fixed_nodef_full_s42_contrastive_full_s42_summary.json`
- `outputs/E1_badnet_rs_fixed_defenses/results/traj_rn50_badnet_rs_fixed_par_full_s42_contrastive_full_s42_summary.json`
- `outputs/E1_badnet_rs_fixed_defenses/results/traj_rn50_badnet_rs_fixed_par_proj_s42_contrastive_lora_s42_summary.json`
- `outputs/E1_clean_purified_controls/results/traj_rn50_clean_invertune_noop_full_s42_contrastive_full_s42_summary.json`
- `outputs/E1_clean_purified_controls/results/traj_rn50_clean_par_full_s42_contrastive_full_s42_summary.json`
- `outputs/E1_projection_existing/results/traj_rn50_align_invertune_proj_s42_contrastive_lora_s42_summary.json`
- `outputs/E1_projection_existing/results/traj_rn50_align_par_proj_s42_contrastive_lora_s42_summary.json`
- `outputs/E1_projection_existing/results/traj_rn50_badclip_par_proj_s42_contrastive_lora_s42_summary.json`
- `outputs/E2_causality_allparam_fix2/rn50_align_par_normal_allparam_s42/rn50_align_par_normal_allparam_s42_summary.json`
- `outputs/E2_causality_allparam_fix2/rn50_align_par_project_oracle_allparam_s42/rn50_align_par_project_oracle_allparam_s42_summary.json`
- `outputs/E2_causality_allparam_fix2/rn50_align_par_proxy_shuffled_allparam_s42/rn50_align_par_proxy_shuffled_allparam_s42_summary.json`
- `outputs/E2_causality_allparam_fix2/rn50_align_par_random_matched_allparam_s42/rn50_align_par_random_matched_allparam_s42_summary.json`
- `outputs/E2_causality_main_fix1/rn50_align_par_normal_s42/rn50_align_par_normal_s42_summary.json`
- `outputs/E3_align_par_main_seed42/downstream/rn50_align_par_compute_matched_cleanft_s42_rebound_full/results/traj_rn50_align_par_compute_matched_cleanft_s42_rebound_full_contrastive_full_s42_summary.json`
- `outputs/E3_align_par_main_seed42/downstream/rn50_align_par_immuneclip_checkpoint_rho_s42_rebound_full/results/traj_rn50_align_par_immuneclip_checkpoint_rho_s42_rebound_full_contrastive_full_s42_summary.json`
- `outputs/E3_align_par_main_seed42/runs/rn50_align_par_compute_matched_cleanft_s42/results/rn50_align_par_compute_matched_cleanft_s42_init_summary.json`
- `outputs/E3_align_par_main_seed42/runs/rn50_align_par_compute_matched_cleanft_s42/results/rn50_align_par_compute_matched_cleanft_s42_summary.json`
- `outputs/E3_align_par_main_seed42/runs/rn50_align_par_immuneclip_checkpoint_rho_s42/results/rn50_align_par_immuneclip_checkpoint_rho_s42_init_summary.json`
- `outputs/E3_align_par_main_seed42/runs/rn50_align_par_immuneclip_checkpoint_rho_s42/results/rn50_align_par_immuneclip_checkpoint_rho_s42_summary.json`
- `outputs/E3_align_par_rescue_seed42/downstream/rn50_align_par_immuneclip_traj_global_working_s42_rebound_full/results/traj_rn50_align_par_immuneclip_traj_global_working_s42_rebound_full_contrastive_full_s42_summary.json`
- `outputs/E3_align_par_rescue_seed42/runs/rn50_align_par_immuneclip_traj_global_working_s42/results/rn50_align_par_immuneclip_traj_global_working_s42_init_summary.json`
- `outputs/E3_align_par_rescue_seed42/runs/rn50_align_par_immuneclip_traj_global_working_s42/results/rn50_align_par_immuneclip_traj_global_working_s42_summary.json`
- `outputs/E3_align_par_rescue_seed42/runs/rn50_align_par_immuneclip_traj_global_working_s42/rn50_align_par_immuneclip_traj_global_working_s42_summary.json`
- `outputs/single_proxy_formal_rebound_summary.json`

## Log File Inventory

- `logs/E1_align_main/master_seed42.log`
- `logs/E1_align_main/nohup_seed42.log`
- `logs/E1_align_main/rn50_align_invertune_full_s42.log`
- `logs/E1_align_main/rn50_align_nodef_full_s42.log`
- `logs/E1_align_main/rn50_align_par_full_s42.log`
- `logs/E1_align_main/rn50_clean_control_full_s42.log`
- `logs/E1_align_main_rerun1/nohup_seed42.log`
- `logs/E1_badclip_invertune/activation_tuning.log`
- `logs/E1_badclip_invertune/baseline_evaluation.log`
- `logs/E1_badclip_invertune/defended_evaluation.log`
- `logs/E1_badclip_invertune/downstream_full.log`
- `logs/E1_badclip_invertune/master_seed42.log`
- `logs/E1_badclip_invertune/trigger_inversion.log`
- `logs/E1_badclip_main/master_seed42.log`
- `logs/E1_badclip_main/rn50_badclip_cleanclip_full_s42.log`
- `logs/E1_badclip_main/rn50_badclip_nodef_full_s42.log`
- `logs/E1_badclip_main/rn50_badclip_par_full_s42.log`
- `logs/E1_badnet_attack/create_badnet_data_seed42.log`
- `logs/E1_badnet_attack/master_seed42.log`
- `logs/E1_badnet_attack/poison_logs/rn50_badnet_random_poison_ep10_s42/output.log`
- `logs/E1_badnet_attack/poison_logs/rn50_badnet_random_poison_ep10_s42/params.txt`
- `logs/E1_badnet_attack/rn50_badnet_random_poison_ep10_s42.log`
- `logs/E1_badnet_attack_middle_targetce/data_prepare.log`
- `logs/E1_badnet_attack_middle_targetce/master_seed42.log`
- `logs/E1_badnet_attack_middle_targetce/poison_logs/rn50_badnet_middle_p10pct_targetce_l10_lr2e6_ep3_s42/output.log`
- `logs/E1_badnet_attack_middle_targetce/poison_logs/rn50_badnet_middle_p10pct_targetce_l10_lr2e6_ep3_s42/params.txt`
- `logs/E1_badnet_attack_middle_targetce/rn50_badnet_middle_p10pct_targetce_l10_lr2e6_ep3_s42.log`
- `logs/E1_badnet_attack_middle_targetce_nohup_seed42.log`
- `logs/E1_badnet_attack_nohup_seed42.log`
- `logs/E1_badnet_attack_strong/create_badnet_random_p5pct_seed42.log`
- `logs/E1_badnet_attack_strong/master_seed42.log`
- `logs/E1_badnet_attack_strong/poison_logs/rn50_badnet_random_p5pct_ep5_s42/output.log`
- `logs/E1_badnet_attack_strong/poison_logs/rn50_badnet_random_p5pct_ep5_s42/params.txt`
- `logs/E1_badnet_attack_strong/rn50_badnet_random_p5pct_ep5_s42.log`
- `logs/E1_badnet_attack_strong_nohup_seed42.log`
- `logs/E1_badnet_attack_targetce/data_prepare.log`
- `logs/E1_badnet_attack_targetce/master_seed42.log`
- `logs/E1_badnet_attack_targetce/poison_logs/rn50_badnet_random_p10pct_targetce_ep3_s42/output.log`
- `logs/E1_badnet_attack_targetce/poison_logs/rn50_badnet_random_p10pct_targetce_ep3_s42/params.txt`
- `logs/E1_badnet_attack_targetce/rn50_badnet_random_p10pct_targetce_ep3_s42.log`
- `logs/E1_badnet_attack_targetce_nohup_seed42.log`
- `logs/E1_badnet_attack_targetce_retry_nohup_seed42.log`
- `logs/E1_badnet_defenses/downstream_rn50_badnet_nodef_full_s42_full.log`
- `logs/E1_badnet_defenses/master_seed42.log`
- `logs/E1_badnet_defenses/par_prepare.log`
- `logs/E1_badnet_defenses/par_train.log`
- `logs/E1_badnet_defenses_nohup_seed42.log`
- `logs/E1_badnet_middle_middle_targetce_defenses/master_seed42.log`
- `logs/E1_badnet_middle_targetce_defenses/master_seed42.log`
- `logs/E1_badnet_middle_targetce_defenses_nohup_seed42.log`
- `logs/E1_badnet_rs_attack/data_prepare.log`
- `logs/E1_badnet_rs_attack/master_seed42.log`
- `logs/E1_badnet_rs_attack/poison_logs/rn50_badnet_rs_p10pct_targetce_l10_lr2e6_ep3_s42/output.log`
- `logs/E1_badnet_rs_attack/poison_logs/rn50_badnet_rs_p10pct_targetce_l10_lr2e6_ep3_s42/params.txt`
- `logs/E1_badnet_rs_attack/rn50_badnet_rs_p10pct_targetce_l10_lr2e6_ep3_s42.log`
- `logs/E1_badnet_rs_attack_nohup_seed42.log`
- `logs/E1_badnet_rs_attack_nohup_seed42.restart2.log`
- `logs/E1_badnet_rs_defenses/master_seed42.log`
- `logs/E1_badnet_rs_defenses_nohup_seed42.log`
- `logs/E1_badnet_rs_fixed_attack/data_prepare.log`
- `logs/E1_badnet_rs_fixed_attack/master_seed42.log`
- `logs/E1_badnet_rs_fixed_attack/poison_logs/rn50_badnet_rs_fixed_p10pct_targetce_l10_lr2e6_ep3_s42/output.log`
- `logs/E1_badnet_rs_fixed_attack/poison_logs/rn50_badnet_rs_fixed_p10pct_targetce_l10_lr2e6_ep3_s42/params.txt`
- `logs/E1_badnet_rs_fixed_attack/rn50_badnet_rs_fixed_p10pct_targetce_l10_lr2e6_ep3_s42.log`
- `logs/E1_badnet_rs_fixed_attack_nohup_seed42.log`
- `logs/E1_badnet_rs_fixed_defenses/cleanclip_logs/cleanclip_badnet_rs_fixed_s42/output.log`
- `logs/E1_badnet_rs_fixed_defenses/cleanclip_logs/cleanclip_badnet_rs_fixed_s42/params.txt`
- `logs/E1_badnet_rs_fixed_defenses/cleanclip_train.log`
- `logs/E1_badnet_rs_fixed_defenses/downstream_rn50_badnet_rs_fixed_cleanclip_full_s42_full.log`
- `logs/E1_badnet_rs_fixed_defenses/downstream_rn50_badnet_rs_fixed_nodef_full_s42_full.log`
- `logs/E1_badnet_rs_fixed_defenses/downstream_rn50_badnet_rs_fixed_par_full_s42_full.log`
- `logs/E1_badnet_rs_fixed_defenses/downstream_rn50_badnet_rs_fixed_par_proj_s42_lora.log`
- `logs/E1_badnet_rs_fixed_defenses/invertune_trigger_inversion.log`
- `logs/E1_badnet_rs_fixed_defenses/master_seed42.log`
- `logs/E1_badnet_rs_fixed_defenses/nohup_retry_par.log`
- `logs/E1_badnet_rs_fixed_defenses/par_prepare.log`
- `logs/E1_badnet_rs_fixed_defenses/par_train.log`
- `logs/E1_badnet_rs_fixed_defenses_nohup_seed42.log`
- `logs/E1_badnet_targetce_defenses/master_seed42.log`
- `logs/E1_badnet_targetce_defenses_nohup_seed42.log`
- `logs/E1_badnet_targetce_defenses_retry_nohup_seed42.log`
- `logs/E1_clean_purified_controls/downstream_rn50_clean_invertune_noop_full_s42.log`
- `logs/E1_clean_purified_controls/downstream_rn50_clean_par_full_s42.log`
- `logs/E1_clean_purified_controls/invertune_clean_baseline.log`
- `logs/E1_clean_purified_controls/invertune_clean_trigger_inversion.log`
- `logs/E1_clean_purified_controls/master_seed42.log`
- `logs/E1_clean_purified_controls/par_clean_train.log`
- `logs/E1_projection_existing/master_seed42.log`
- `logs/E1_projection_existing/rn50_align_invertune_proj_s42.log`
- `logs/E1_projection_existing/rn50_align_par_proj_s42.log`
- `logs/E1_projection_existing/rn50_badclip_par_proj_s42.log`
- `logs/E2_causality_allparam_fix2/master_seed42.log`
- `logs/E2_causality_allparam_fix2/nohup_seed42.log`
- `logs/E2_causality_allparam_fix2/rn50_align_par_normal_allparam_s42.log`
- `logs/E2_causality_allparam_fix2/rn50_align_par_project_oracle_allparam_s42.log`
- `logs/E2_causality_allparam_fix2/rn50_align_par_proxy_shuffled_allparam_s42.log`
- `logs/E2_causality_allparam_fix2/rn50_align_par_random_matched_allparam_s42.log`
- `logs/E2_causality_main/master_seed42.log`
- `logs/E2_causality_main/nohup_seed42.log`
- `logs/E2_causality_main/rn50_align_par_normal_s42.log`
- `logs/E2_causality_main_fix1/master_seed42.log`
- `logs/E2_causality_main_fix1/nohup_seed42.log`
- `logs/E2_causality_main_fix1/rn50_align_par_normal_s42.log`
- `logs/E2_causality_main_fix1/rn50_align_par_project_oracle_harmful_s42.log`
- `logs/E3_align_par_main_seed42/downstream_rn50_align_par_compute_matched_cleanft_s42_rebound_full.log`
- `logs/E3_align_par_main_seed42/downstream_rn50_align_par_immuneclip_checkpoint_rho_s42_rebound_full.log`
- `logs/E3_align_par_main_seed42/master_seed42.log`
- `logs/E3_align_par_main_seed42/train_rn50_align_par_compute_matched_cleanft_s42.log`
- `logs/E3_align_par_main_seed42/train_rn50_align_par_immuneclip_checkpoint_rho_s42.log`
- `logs/E3_align_par_main_seed42_nohup.log`
- `logs/E3_align_par_rescue_seed42/downstream_server1.log`
- `logs/E3_align_par_rescue_seed42/master_seed42.log`
- `logs/E3_align_par_rescue_seed42/nohup.log`
- `logs/E3_align_par_rescue_seed42/train_rn50_align_par_immuneclip_traj_global_working_s42.log`
