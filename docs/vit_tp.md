# DeiT-S with Tensor Parallelism (`vit_tp_exp`)

`tinyexp/examples/vit_tp_exp.py` trains and evaluates **DeiT-S** with **tensor
parallelism** (TP), implemented against the official DeiT sources line by line
(对拍): only formatting-level changes are allowed, and every deviation from upstream is
listed below and proven numerically equivalent by tests. This page is the complete
plan-and-record home for the example: ported sources, sanctioned deviations, usage,
and the four-layer cross-check record (numerics, eval accuracy, full-training
accuracy, throughput).

## Why DeiT-S (and not DeiT-Ti)

Tensor parallelism shards attention on **head boundaries**, so `num_heads` must be
divisible by the TP size. DeiT-Ti has 3 heads — structurally unusable at TP=2. DeiT-S has
6 heads (3 per rank at TP=2), 22,050,664 parameters, and the official 300-epoch recipe
reaches **79.8% top-1** on ImageNet val, which is the accuracy cross-check target.

## Ported sources (pinned)

| Component | Source |
| --- | --- |
| Model (`VisionTransformer`/`Block`/`Attention`/`Mlp`/`PatchEmbed`) | timm `vision_transformer` implementation configured for timm 0.3.2 DeiT behavior; `Attention` is adapted locally to split QKV for TP |
| Factory hyper-parameters, checkpoint URL | facebookresearch/deit `models.py` @ `7e160fe4` |
| Train/eval loops, recipe defaults, transforms, `RASampler`, `MetricLogger` | deit `engine.py`/`main.py`/`datasets.py`/`samplers.py`/`utils.py` (same commit) |
| Data/loss/optimizer helpers (`Mixup`, `create_transform`, `NativeScaler`, `create_scheduler`, …) | timm >= 1.0 (same algorithms) |

TP itself is `torch.distributed.tensor.parallel` (`ColwiseParallel`/`RowwiseParallel`)
wrapped by `TPAccelerator`, following the classic Megatron block layout: colwise on
`attn.q/k/v` + `mlp.fc1`, rowwise on `attn.proj` + `mlp.fc2`, one all-reduce per block
pass. Attention runs per-head locally with no communication inside, using PyTorch's
`scaled_dot_product_attention` (FlashAttention backend on supported CUDA devices).

## Sanctioned deviations (each covered by a test)

- **D1** — the official fused `attn.qkv` Linear is split into `q`/`k`/`v` so colwise
  sharding lands on head boundaries. `convert_fused_qkv_to_split` /
  `convert_split_qkv_to_fused` translate weights; checkpoints always store the official
  fused layout and load official DeiT checkpoints directly.
- **D2** — `Attention.forward` reshapes use `-1` (local-shape aware), as torch's
  `use_local_output=True` TP pattern requires; at tp=1 the shapes resolve to exactly the
  official ones.
- **D3** — `mode=bench` is an added capability (synthetic throughput/memory check); it
  does not touch the training semantics.
- **D4** — EMA (official default on) is disabled at tp>1: an EMA copy holds full weights
  and per-step gathering from DTensors is expensive.
- **D5/D6** — under TP all ranks consume identical batches and share one RNG seed (TP
  collectives require identical inputs; the official `seed + rank` is the DDP
  data-sharding context).
- **D7** — optional Redis byte cache for the training set
  (`redis_cfg.redis_cache_enabled`, default **off**; ray-managed Redis, no external
  wrapper needed): the cache stores raw JPEG bytes verbatim, so samples are
  bit-identical to `datasets.ImageFolder`; the first epoch fills the cache from disk,
  later epochs read from memory. `safe_set` failures (e.g. maxmemory full) fall back to
  disk reads — correctness never depends on the cache.

## Usage

Full recipe (2 GPUs, TP=2, official defaults: 300 epochs, AdamW wd 0.05, cosine with
5-epoch warmup, mixup 0.8/cutmix 1.0, repeated augmentation):

```bash
python -m tinyexp.examples.vit_tp_exp
```

Eval-only against the official checkpoint (accuracy cross-check, expects 79.8 ± 0.1):

```bash
python -m tinyexp.examples.vit_tp_exp mode=eval \
    module_cfg.pretrained_from=https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
```

Throughput / memory benchmark (compare against DDP by overriding the accelerator):

```bash
python -m tinyexp.examples.vit_tp_exp mode=bench
python -m tinyexp.examples.vit_tp_exp mode=bench accelerator_cfg.accelerator=ddp
```

CPU smoke test (no ImageNet, ray 2 workers, gloo TP=2):

```bash
python -m tinyexp.examples.vit_tp_exp dataloader_cfg.fake_data=true dataloader_cfg.input_size=32 \
    module_cfg.img_size=32 module_cfg.num_classes=10 epochs=1 max_train_steps=4 \
    ray_cfg.ray_num_gpus_per_worker=0.0 ray_cfg.ray_num_cpus_per_worker=4
```

Full 300-epoch training on many GPUs (e.g. 8×H200/H800 single node, DDP — DeiT-S has
6 heads and cannot shard over TP=8), with the Redis byte cache for slow shared
filesystems (D7; the ray mixin starts the Redis shards itself):

```bash
python -m tinyexp.examples.vit_tp_exp accelerator_cfg.accelerator=ddp \
    ray_cfg.ray_num_worker=8 redis_cfg.redis_cache_enabled=true redis_cfg.redis_cache_max_memory=300
```

Recipe note (global batch / lr): the official linear rule `lr × global_batch / 512`
is applied verbatim — the default 2×256=512 gives lr 5e-4 (the same scaling point as
the official 1024 @ 1e-3), and the 8×256=2048 H200 run below used lr 2e-3.

## Data preparation

ImageNet-1k in the standard `ImageFolder` layout, located via `IMAGENET_HOME` or
`dataloader_cfg.data_root` (default `./data/imagenet/`):

```
<root>/train/<wnid>/*.JPEG    # 1.28M images, 1000 class dirs (full training)
<root>/val/<wnid>/*.JPEG      # 50k images (eval-only cross-check)
```

## Cross-check record

Four layers, each with a hard pass criterion; all hit.

- **L1 — numerical equivalence** (CI, CPU+gloo): TP=2 forward logits and one-step
  gradients match the single-process run at fp32 `allclose(atol=1e-5)`
  (`tests/examples/test_vit_tp_exp_tp.py`), and the split-qkv forward matches the
  fused official one at `atol=1e-6`.
- **L2 — eval accuracy** (official checkpoint, full 50k val): **79.82% top-1 /
  94.95% top-5 at tp=1**, **79.81% / 94.94% at tp=2** (official reference 79.8;
  tp=1 vs tp=2 differ by 0.01%).
- **L3 — full-training accuracy** (300-epoch official recipe from scratch, 8×H200
  DDP via rjob, global batch 2048 / lr 2e-3, Redis byte cache on): **best top-1
  79.84% / top-5 94.99%** in 17h56m (~3.5 min/epoch incl. full-val each epoch) —
  paper reports 79.8 / ~95.0. Trajectory: ep45 58.6 → ep113 69.7 → ep181 74.1 →
  ep249 78.4 → ep299 79.84. Artifacts: `output/vit_tp_l3_h200/{last,best}.ckpt`
  (official fused-qkv layout), per-epoch `log.txt`.
- **L4 — throughput/memory** (bench mode, table below).

Two engine hardenings landed with this example, each held by a regression test:
`DDPAccelerator.reduce_sum` stages CPU tensors through the device so nccl
metric-sync works at epoch boundaries, and `store_and_run_exp` resolves the
canonical importable twin of a `python -m` `__main__` class so ray ships it by
reference (robust against cluster agents that replace `builtins.print` after
import).

Redis cache behavior measured on the L3 run: `RASampler(num_repeats=3)` visits
~1/3 unique indices per epoch, so the cache fills over ~3 epochs to the full
1,281,167-key train set (6 shards × 213,528); once warm, data time drops to
~0.6 ms/step and steps run at ~0.25 s.

## Measured on 2× RTX 4080 (DeiT-S, 224, AMP)

| Config | batch/rank | step latency | effective img/s | peak mem/rank |
| --- | --- | --- | --- | --- |
| DDP=2 | 256 (its ceiling; 288 OOM) | 294.7 ms | 1737 | 13.44 GB |
| TP=2 | 256 | 841.0 ms | 304 | **8.61 GB** |
| TP=2 | 384 (its ceiling; 512 OOM) | 1262.5 ms | 304 | 12.87 GB |

Honest reading: at DeiT-S scale TP is **not** a throughput win (communication dominates;
~17% of DDP). What TP buys is per-device memory — a 1.5× larger per-rank batch ceiling
(384 vs 256) on the same 16 GB cards. The correctness gate is met: evaluating the official checkpoint on the full 50k
ImageNet val yields **79.82% top-1 / 94.95% top-5 at tp=1** and **79.81% / 94.94% at
tp=2** (official reference: 79.8%); numerical equivalence of TP=2 vs single-rank
forward/backward is asserted in CI (fp32 `allclose`, `tests/examples/test_vit_tp_exp_tp.py`).
