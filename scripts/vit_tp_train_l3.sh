#!/usr/bin/env bash
# L3 full training (docs/vit_tp.md): DeiT-S, 8-GPU DDP on a single H200/H800 node.
#
# Ray owns the whole lifecycle: launcher=ray spawns 8 GPU workers, and the ray
# mixin starts standalone Redis itself when redis_cache_enabled=true (no
# tinyexp-run-with-redis wrapper for ray launches). The Redis byte cache
# (docs/vit_tp.md D7) fills from JuiceFS during epoch 1; later epochs read from
# memory. Global batch 8x256=2048; OptimizerCfg applies the official linear
# rule automatically (lr -> 5e-4 * 2048/512 = 2e-3).
#
# Smoke first, e.g.:
#   zane_launch ... -- bash scripts/vit_tp_train_l3.sh exp_name=vit_tp_l3_smoke max_train_steps=20
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export IMAGENET_HOME=/mnt/step2-alignment-jfs/zhanghan/data/imagenet
# The repo's ./output symlink targets this jfs path; keep it resolvable for hydra RunDir.
mkdir -p /mnt/jfs-zane-research/outputs/tinyexp

command -v redis-server >/dev/null || { echo "redis-server not found in PATH; required by the ray-managed redis cache" >&2; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

python - <<'EOF'
import timm
import torch

print("torch", torch.__version__, "| timm", timm.__version__)
assert torch.cuda.is_available(), "no CUDA device visible in rjob"
print("visible GPUs:", torch.cuda.device_count())
EOF

test -d "$IMAGENET_HOME/train"
test -d "$IMAGENET_HOME/val"

python -m tinyexp.examples.vit_tp_exp \
    accelerator_cfg.accelerator=ddp \
    ray_cfg.ray_num_worker=8 \
    redis_cfg.redis_cache_enabled=true \
    redis_cfg.redis_cache_max_memory=300 \
    exp_name=vit_tp_l3_h800 \
    "$@"
