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
