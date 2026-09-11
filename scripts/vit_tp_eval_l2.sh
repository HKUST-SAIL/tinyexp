#!/usr/bin/env bash
# docs/vit_tp.md L2 accuracy cross-check, eval-only, 1 GPU (tp=1).
# Official DeiT-S checkpoint + full ImageNet val; expect top-1 79.8 ±0.1
# (measured 79.82 / 94.95 on 2x4080, tp=1; record in docs/vit_tp.md).
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export IMAGENET_HOME=/mnt/step2-alignment-jfs/zhanghan/data/imagenet
CKPT="$PWD/data/deit_small_patch16_224-cd65a155.pth"

# The repo's ./output symlink targets this jfs path; make sure it resolves to an
# existing dir so hydra's RunDir mkdir does not hit a dangling symlink.
mkdir -p /mnt/jfs-zane-research/outputs/tinyexp

nvidia-smi || true

python - <<'EOF'
import timm
import torch

print("torch", torch.__version__, "| timm", timm.__version__)
assert torch.cuda.is_available(), "no CUDA device visible in rjob"
EOF

# Fail fast if the JuiceFS ImageNet mount is not attached in this container.
test -d "$IMAGENET_HOME/val"

python -m tinyexp.examples.vit_tp_exp \
    mode=eval \
    launcher=mp \
    module_cfg.pretrained_from="$CKPT" \
    "$@"
