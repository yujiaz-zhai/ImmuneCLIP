#!/usr/bin/env python3
"""统一 ASR/CA 评估入口。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from clip_eval import eval_asr_ca
from config import CKPT_CLEAN, CKPT_POISONED, LOG_ROOT, RESULT_ROOT


def main():
    parser = argparse.ArgumentParser(description="ImmuneCLIP eval ASR/CA")
    parser.add_argument("--ckpt", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--subset", type=int, default=5000, help="ImageNet 评估子集大小，0=全量")
    parser.add_argument("--tag", type=str, default="", help="实验标签")
    parser.add_argument("--out", type=str, default=None, help="结果 json 输出路径")
    args = parser.parse_args()

    subset = None if args.subset <= 0 else args.subset
    metrics = eval_asr_ca(args.ckpt, device=args.device, subset=subset)

    record = {
        "timestamp": datetime.now().isoformat(),
        "tag": args.tag,
        "checkpoint": args.ckpt,
        "subset": subset,
        **metrics,
    }
    print(json.dumps(record, indent=2))

    out = args.out or os.path.join(
        RESULT_ROOT, f"eval_{args.tag or 'run'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved: {out}")
    return record


if __name__ == "__main__":
    main()
