import json
import logging
import os

import pandas as pd


def _load_train_csv(path):
    df = pd.read_csv(path)
    unnamed = [column for column in df.columns if column.startswith("Unnamed:")]
    if unnamed:
        df = df.rename(columns={unnamed[0]: "orig_idx"})
    if "orig_idx" not in df.columns:
        df["orig_idx"] = list(range(len(df)))
    return df


def _resolve_image_paths(df, image_key, root):
    image_series = df[image_key].astype(str)
    df[image_key] = image_series.map(
        lambda value: value if os.path.isabs(value) else os.path.abspath(os.path.join(root, value))
    )
    return df


def _keep_columns(df, image_key, caption_key):
    keep = [image_key, caption_key]
    extra = [column for column in ["orig_idx"] if column in df.columns]
    return df[keep + extra].copy()


def prepare_sbl_stage_csvs(options):
    poisoned_path = options.train_data
    clean_path = options.clean_train_data
    if clean_path is None:
        clean_path = os.path.join(os.path.dirname(poisoned_path), "train.csv")

    output_dir = os.path.join(options.log_dir_path, "sbl_data")
    os.makedirs(output_dir, exist_ok=True)

    step0_path = os.path.join(output_dir, "sbl_step0_mixed.csv")
    step1_path = os.path.join(output_dir, "sbl_step1_clean.csv")
    summary_path = os.path.join(output_dir, "split_summary.json")

    if os.path.exists(step0_path) and os.path.exists(step1_path) and os.path.exists(summary_path):
        with open(summary_path, "r") as file:
            summary = json.load(file)
        logging.info(f"Reusing existing SBL split under {output_dir}")
        return {
            "step0_train_data": step0_path,
            "step1_train_data": step1_path,
            "summary": summary,
        }

    poisoned_root = os.path.dirname(poisoned_path)
    clean_root = os.path.dirname(clean_path)

    poisoned_df = _keep_columns(_load_train_csv(poisoned_path), options.image_key, options.caption_key)
    clean_df = _keep_columns(_load_train_csv(clean_path), options.image_key, options.caption_key)
    poisoned_df = _resolve_image_paths(poisoned_df, options.image_key, poisoned_root)
    clean_df = _resolve_image_paths(clean_df, options.image_key, clean_root)

    poison_mask = poisoned_df[options.image_key].astype(str).str.contains("backdoor_images_")
    poisoned_rows = poisoned_df[poison_mask].copy()
    poisoned_orig_idx = set(poisoned_rows["orig_idx"].tolist())

    clean_pool = clean_df[~clean_df["orig_idx"].isin(poisoned_orig_idx)].copy()

    total_samples = len(clean_df)
    step0_target = int(total_samples * options.sbl_mixed_portion)
    step1_target = int(total_samples * options.sbl_clean_portion)
    step0_clean_target = max(0, step0_target - len(poisoned_rows))

    if step0_clean_target + step1_target > len(clean_pool):
        raise ValueError("SBL split exceeds available clean pool. Reduce mixed/clean portions.")

    step0_clean = clean_pool.sample(n=step0_clean_target, random_state=options.sbl_seed)
    remaining_clean = clean_pool[~clean_pool["orig_idx"].isin(step0_clean["orig_idx"])].copy()
    step1_clean = remaining_clean.sample(n=step1_target, random_state=options.sbl_seed + 1)

    step0_df = pd.concat(
        [
            poisoned_rows[[options.image_key, options.caption_key]],
            step0_clean[[options.image_key, options.caption_key]],
        ],
        ignore_index=True,
    )
    step1_df = step1_clean[[options.image_key, options.caption_key]].reset_index(drop=True)

    step0_df.to_csv(step0_path, index=False)
    step1_df.to_csv(step1_path, index=False)

    summary = {
        "poisoned_path": poisoned_path,
        "clean_path": clean_path,
        "step0_train_data": step0_path,
        "step1_train_data": step1_path,
        "step0_total": len(step0_df),
        "step0_poisoned": int(len(poisoned_rows)),
        "step0_clean": int(len(step0_clean)),
        "step1_clean": int(len(step1_df)),
        "unused_clean": int(len(remaining_clean) - len(step1_df)),
        "mixed_portion": options.sbl_mixed_portion,
        "clean_portion": options.sbl_clean_portion,
        "seed": options.sbl_seed,
    }

    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    logging.info(
        "Prepared SBL split: "
        f"step0={summary['step0_total']} (poisoned={summary['step0_poisoned']}, clean={summary['step0_clean']}), "
        f"step1={summary['step1_clean']}"
    )
    return {
        "step0_train_data": step0_path,
        "step1_train_data": step1_path,
        "summary": summary,
    }
