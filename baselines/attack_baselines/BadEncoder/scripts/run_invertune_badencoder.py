import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLIP_MODEL = "/root/autodl-tmp/checkpoints/clip-clean-pretrained/RN50.pt"
CC3M_ROOT = "/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K"
CC3M_CSV = f"{CC3M_ROOT}/train.csv"
IMAGENET_ROOT = "/root/autodl-tmp/datasets/imagenet1k_badclip/validation"
IMAGENET_CLASSES = "/root/autodl-tmp/experiments/invertune/data/classes.py"
BANANA_REFERENCE = f"{IMAGENET_ROOT}/images/ILSVRC2012_val_00047701.JPEG"
TRIGGER = str(PROJECT_ROOT / "trigger/trigger_pt_white_173_50_ap_replace.npz")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "output/CLIP/invertune_badencoder_banana")
DEFAULT_LOG = str(PROJECT_ROOT / "log/invertune_badencoder_banana.log")


def run(command, log_path):
    command_text = "$ " + " ".join(shlex.quote(part) for part in command)
    print(command_text, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log_file:
        log_file.write((command_text + "\n").encode())
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        while True:
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log_file.write(chunk)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the paper-configured InverTune BadEncoder baseline."
    )
    parser.add_argument("--stage", choices=("train", "eval", "all"), default="all")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--max_shadow_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--log_file", default=DEFAULT_LOG)
    parser.add_argument("--resume", default="")
    parser.add_argument("--checkpoint_every", default=1, type=int)
    parser.add_argument("--disable_amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file)
    checkpoint = output_dir / f"model_{args.epochs}.pth"

    if args.stage in ("train", "all"):
        train_command = [
            sys.executable,
            "-u",
            "badencoder.py",
            "--lr",
            "1e-6",
            "--batch_size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--lambda1",
            "1.0",
            "--lambda2",
            "1.0",
            "--shadow_dataset",
            "cc3m",
            "--shadow_data_root",
            CC3M_ROOT,
            "--shadow_csv",
            CC3M_CSV,
            "--pretrained_encoder",
            CLIP_MODEL,
            "--encoder_usage_info",
            "CLIP",
            "--reference_file",
            BANANA_REFERENCE,
            "--trigger_file",
            TRIGGER,
            "--target_label",
            "954",
            "--target_word",
            "banana",
            "--results_dir",
            str(output_dir),
            "--num_workers",
            str(args.num_workers),
            "--gpu",
            str(args.gpu),
            "--checkpoint_every",
            str(args.checkpoint_every),
        ]
        if args.max_shadow_samples:
            train_command.extend(
                ["--max_shadow_samples", str(args.max_shadow_samples)]
            )
        if args.resume:
            train_command.extend(["--resume", args.resume])
        if args.disable_amp:
            train_command.append("--disable_amp")
        run(train_command, log_path)

    if args.stage in ("eval", "all"):
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint {checkpoint}. Run --stage train first or "
                "set --epochs to the checkpoint epoch."
            )
        eval_command = [
            sys.executable,
            "-u",
            "zero_shot_imagenet.py",
            "--data_root",
            IMAGENET_ROOT,
            "--classes_file",
            IMAGENET_CLASSES,
            "--encoder",
            str(checkpoint),
            "--clip_model",
            CLIP_MODEL,
            "--trigger_file",
            TRIGGER,
            "--target_label",
            "954",
            "--target_word",
            "banana",
            "--batch_size",
            str(args.eval_batch_size),
            "--num_workers",
            str(args.num_workers),
            "--gpu",
            str(args.gpu),
            "--output",
            str(output_dir / "imagenet_zero_shot_metrics.json"),
        ]
        if args.max_eval_samples:
            eval_command.extend(["--max_samples", str(args.max_eval_samples)])
        run(eval_command, log_path)


if __name__ == "__main__":
    main()
