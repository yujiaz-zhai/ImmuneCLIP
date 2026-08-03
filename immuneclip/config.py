"""ImmuneCLIP Week1 路径与默认超参（复用 aaai 环境）。"""
import os

USENIX_ROOT = "/root/workspace/usenix"
BADCLIP_ROOT = os.path.join(USENIX_ROOT, "baselines/BadCLIP")
EXP_ROOT = os.environ.get("IMMUNECLIP_EXP_ROOT", "/root/autodl-tmp/experiments/immuneclip/week1")
LOG_ROOT = os.path.join(EXP_ROOT, "logs")
RESULT_ROOT = os.path.join(EXP_ROOT, "results")
CKPT_ROOT = os.path.join(EXP_ROOT, "checkpoints")

# Checkpoints
CKPT_CLEAN = "/root/autodl-tmp/checkpoints/clip-clean-pretrained/RN50.pt"
CKPT_POISONED = (
    "/root/autodl-tmp/experiments/badclip_sbl/baseline_1_1_readme_strict/"
    "logs/nodefence_badCLIP/checkpoints/epoch_10.pt"
)
CKPT_PAR_CLEANED = os.path.join(CKPT_ROOT, "par_cleaned_rn50.pt")  # 需训练或下载
CKPT_CLEANCLIP = os.path.join(CKPT_ROOT, "cleanclip_badclip_rn50.pt")  # Day5

# Data
IMAGENET_VAL_DIR = "/root/autodl-tmp/datasets/imagenet1k_badclip/validation"
IMAGENET_VAL_IMAGES = os.path.join(IMAGENET_VAL_DIR, "images")
CC3M_TRAIN_CSV = "/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K/train.csv"
CC3M_IMAGES = "/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K/images"

# Attack
TARGET_LABEL = "banana"
PATCH_NAME = "opti_patches/badCLIP.jpg"
PATCH_TYPE = "ours_tnature"
PATCH_LOCATION = "middle"
PATCH_SIZE = 16
MODEL_NAME = "RN50"

# Eval
EVAL_BATCH_SIZE = 64
EVAL_NUM_WORKERS = 4
IMAGENET_EVAL_SUBSET = 5000  # Week1 快测子集；设为 None 用全量 50k

os.makedirs(LOG_ROOT, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)
os.makedirs(CKPT_ROOT, exist_ok=True)
