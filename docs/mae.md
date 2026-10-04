# MAE ViT-B Pretraining, Linear Probing, and Evaluation

`tinyexp/examples/mae_exp.py` ports [facebookresearch/mae](https://github.com/facebookresearch/mae)
line by line (对拍): only formatting-level changes are allowed, and every deviation from
upstream is listed below and pinned by tests. It provides **MAE ViT-B pretraining** (the
masked-autoencoder algorithm) and **evaluation** of the officially released fine-tuned
checkpoint on the ImageNet val set.

`tinyexp/examples/mae_linprobe_exp.py` adds **linear-probe training and evaluation** as a
subclass of `MaeExp`, reusing its model, data, launcher, logging, and checkpoint plumbing.
The linear-probe entry is **two-GPU smoke-verified** (record below), but full 90-epoch
ImageNet accuracy is unvalidated. The later cross-check section records `mae_exp` results.
Finetuning training is still not implemented.

## Ported sources (pinned)

All mae sources at commit `efb2a8062c206524e35e47d04501ed4f544c0ae8` (public HEAD):

| Source | What is ported |
| --- | --- |
| `models_mae.py` | `MaskedAutoencoderViT` (patchify, random masking, encoder/decoder, masked MSE with `norm_pix_loss`) and `mae_vit_base_patch16_dec512d8b` factory |
| `main_pretrain.py` | seeding (`seed + rank`), `cudnn.benchmark`, train transform ("simple augmentation"), optimizer wiring, per-epoch log stats, resume |
| `engine_pretrain.py` | `train_one_epoch`: per-iteration LR step, autocast forward `model(samples, mask_ratio)`, loss scaler, gradient accumulation, NaN stop |
| `util/lr_sched.py` | `adjust_learning_rate` (linear warmup + half-cycle cosine, `lr_scale` aware) |
| timm `optim_factory` | `add_weight_decay` grouping (timm >= 1.0 spelling: `timm.optim.param_groups_weight_decay`) |
| `models_vit.py` | `VisionTransformer` (global-pool classifier) and `vit_base_patch16` factory |
| `main_finetune.py` | eval slice: val dataset/sampler wiring, checkpoint loading (finetune branch) |
| `util/misc.py` | checkpoint loading (resume branch: URL/local path, `checkpoint['model']`) |
| `util/pos_embed.py` | `get_2d_sincos_pos_embed`, `interpolate_pos_embed` |
| `util/datasets.py` | `build_transform(is_train=False)` eval branch |
| `engine_finetune.py` | `evaluate` (the `_evaluate` method) |
| `util/misc.py` + deit `utils.py` | `SmoothedValue`/`MetricLogger`/`_set_cudnn_benchmark` — reused from `vit_tp_exp` (same detr-lineage port), not duplicated here |
| resnet_exp pattern | `RedisCachedImageFolder` train-set byte cache — reused from `vit_tp_exp` (both wrap the same resnet_exp cache); see D11 |

The timm base is **>= 1.0** here (upstream pins `timm==0.3.2`), configured to reproduce
the 0.3.2 parameter layout and initialization (D1/D2 below).

## Sanctioned deviations (each covered by a test or a recorded check)

- **D1 — timm >= 1.0 base.** The `Block`/`PatchEmbed`/`VisionTransformer` layers come
  from timm 1.x instead of timm 0.3.2. `qk_scale=None` (passed positionally by upstream in
  both `models_mae.py` and `models_vit.py`) is dropped because timm 1.x `Block` has no
  such parameter; the value equals the default `head_dim**-0.5` scale.
  `weight_init="skip"` plus the local `_init_old_weights` reproduces the timm 0.3.2
  default initialization (`trunc_normal_(std=0.02)` etc., the same helper as
  `vit_tp_exp`).
- **D2 — timm 1.x spelling of the global-pool classifier.** The official bool
  `global_pool` is translated to timm's `"avg"`/`"token"`; `"avg"` makes timm build
  `fc_norm` and set `norm = nn.Identity()`, which is byte-equivalent to the official
  `del self.norm` (state-dict layout: `fc_norm.*` present, no `norm.*`). Because timm
  1.x `forward()` passes attention kwargs into `forward_features`, the official one-line
  head call is spelled out in `forward()`. The unit tests prove the composed forward is
  bit-identical to timm's native `forward_features` + `forward_head` stages (max diff
  0.0), which is the numerical license for the timm >= 1.0 base. The pool branch is
  compared against `"avg"` (not truthiness) because timm stores a string and `"token"`
  is also truthy.
- **D3 — merged checkpoint loading.** Official `--resume` (used by the FINETUNE.md
  sanity command) and `--finetune` are one loader here: URL/local path, drop a
  shape-mismatched `head.*`, `interpolate_pos_embed`, `load_state_dict(strict=False)`.
  The official missing-key assert (written for pretraining checkpoints, which miss
  exactly the head) is relaxed to a subset check so the finetuned checkpoint (missing
  nothing) also passes, and the manual head init (`trunc_normal_(std=2e-5)`) only runs
  when head keys were actually dropped. For the finetuned ViT-B checkpoint both official
  paths assign identical weights. The loader serves both model families (autoencoder
  warm-start and classifier eval).
- **D4 — engine plumbing.** Logger is loguru (rank-filtered), `MetricLogger` sync uses
  the accelerator's `reduce`, the deprecated `torch.cuda.amp.autocast()` is spelled
  `torch.amp.autocast(device.type, enabled=device.type == "cuda")` (identical on CUDA,
  valid on CPU), and the training-loop `torch.cuda.synchronize()` is guarded by
  `device.type == "cuda"` so CPU smoke runs work. `SmoothedValue`/`MetricLogger`/
  `_set_cudnn_benchmark` are imported from `vit_tp_exp` instead of duplicated: both
  examples port the same detr-lineage upstream, and the shared implementation already
  carries these engine adaptations.
- **D5 — bicubic spelling.** `transforms.InterpolationMode.BICUBIC` instead of
  `PIL.Image.BICUBIC` / `interpolation=3` (the torchvision v1 transform API maps them
  1:1).
- **D6 — `fake_data` smoke path.** A synthetic `TensorDataset` (random weights) so both
  the training and eval loops can run on CI without ImageNet or a download. The
  transforms themselves are pinned by their own unit tests since the smoke path
  bypasses them.
- **D7 — `np.float` removal.** Upstream `get_1d_sincos_pos_embed_from_grid` uses
  `np.arange(..., dtype=np.float)`, an alias removed in numpy >= 1.24; the port spells
  it `float` (bit-identical dtype).
- **D8 — checkpoint format and cadence.** Official pretraining writes
  `checkpoint-{epoch}.pth` (`model`/`optimizer`/`epoch`/`scaler`/`args`) every 20 epochs
  plus the last; the port writes the tinyexp `last.ckpt` (`model_state_dict`/
  `optimizer_state_dict`/`scaler_state_dict`/`epoch`/`global_step`, atomic
  temp-file + rename) **every epoch** via `CheckpointCfgMixin`, and `resume_from`
  restores model/optimizer/scaler plus continues at `epoch + 1` — the same resume
  semantics as official `util/misc.load_model`. Run artifacts land in
  `output/mae_exp/` (`mae_train.log` loguru log, `log.txt` per-epoch JSON stats).
- **D9 — timm >= 1.0 API renames in the training path.**
  `optim_factory.add_weight_decay` → `timm.optim.param_groups_weight_decay` (same
  grouping: no wd for bias/norm `ndim<=1`, frozen pos embeds skipped), and
  `NativeScaler(..., update_grad=...)` → `need_update=...` (renamed kwarg, identical
  semantics). `NativeScaler(device=accelerator.device.type)` is spelled explicitly so
  the scaler auto-disables on CPU smoke runs.
- **D10 — smoke step cap.** `max_train_steps` (resnet_exp precedent; official has no
  step cap) stops the training loop mid-epoch once the running step count is reached.
  Every rank computes the same count, so the early stop stays DDP-safe.
- **D11 — Redis train-set byte cache (resnet_exp pattern, default on).** The train split
  is served through `RedisCachedImageFolder` (reused from `vit_tp_exp`, the same wrapper
  resnet_exp uses): the cache stores the exact raw file bytes and the decode/transform
  pipeline is unchanged, so samples are bit-identical to `datasets.ImageFolder`. First
  epoch fills the cache from disk (misses), later epochs read from Redis (hits). The val
  split is read once per run and stays uncached. The Ray driver starts the Redis shards
  automatically (ports 7000-7005 by default, stopped with the run); disable with
  `redis_cfg.redis_cache_enabled=false`.
- **D12 — periodic val reconstruction-loss monitor (health check).** Official
  `main_pretrain.py` has no evaluation at all during pretraining; for long cluster
  runs this port additionally evaluates the masked-reconstruction loss on the val
  split every `eval_every_n_epochs` (=10) epochs, after the per-epoch checkpoint (the
  same checkpoint→evaluate order as official `main_finetune.py`). It is monitor-only —
  `model.eval()` + no-grad + the training `mask_ratio`, does not touch the training
  math, optimizer, or schedule — and the value lands in the per-epoch JSON
  (`val_loss`) and wandb next to `train_loss`, so divergence/data problems surface
  within 10 epochs instead of after the full run. `<=0` disables it; a run without a
  val split skips it with one warning (train-only data still works).
  **Scope note**: reconstruction loss is a health check, not a representation-quality
  metric — pixel MSE is dominated by high-frequency detail and is known not to track
  downstream performance (which is why the official protocol is linear probing and the
  paper's ablations are decided by linprobe, not recon loss). The quality progress bar
  is the k-NN monitor below.
- **D13 — periodic k-NN top-1 monitor (representation-quality progress bar).** Every
  `knn_cfg.knn_every_n_epochs` (=50) epochs, extract the official linprobe feature
  (encoder cls token at `mask_ratio=0` — `main_linprobe.py` probes the cls token),
  build a feature bank from `knn_bank_size` (=51200) stride-sampled train images under
  the deterministic eval transform (Redis bytes reused when the cache is on), and
  classify `knn_val_size` (=5000) stride-sampled val images by cosine-similarity
  `knn_topk` (=20)-NN majority vote (DINO-style monitor). Every rank replicates
  bank+query, so no collectives are involved and the number is exact; cost is a few
  minutes every 50 epochs (<1% of a run). The value lands in the per-epoch JSON
  (`knn_top1`) next to `train_loss`/`val_loss` and in wandb. It **correlates with, but
  is not, linear probing** — expect it well below the official 67.8% linprobe for
  ViT-B; what matters is its trend. `knn_every_n_epochs<=0` disables it.

## Usage

Pretrain on 8 GPUs — the complete official recipe in **one command** (Ray workers, the
default launcher; recipe defaults are the official PRETRAIN.md values: batch 64/GPU,
mask ratio 0.75, 800 epochs, 40 warmup epochs, blr 1.5e-4 with the
`lr = blr * eff_batch / 256` linear rule, weight decay 0.05, AdamW betas (0.9, 0.95),
`norm_pix_loss`). `accum_iter` defaults to 8 so an 8-GPU host holds the official
effective batch 4096 (`64 * 8 * 8`, lr 2.4e-3) — identical optimizer math to the
official 64-GPU run, at the same throughput (accumulation only groups updates). The
Redis train-set cache is on by default (resnet_exp style) — the Ray driver starts the
shards automatically, the first epoch fills the cache and later epochs read bytes from
memory instead of the shared FS:

```bash
export IMAGENET_HOME=/path/to/imagenet
python -m tinyexp.examples.mae_exp ray_cfg.ray_num_worker=8
```

On other cluster sizes, scale `accum_iter` so `batch * accum_iter * world = 4096`
(64 GPUs: `accum_iter=1`). Cache knobs: `redis_cfg.redis_cache_enabled=false` turns it
off; `redis_cfg.redis_cache_max_memory` (GB, default 160 ≈ one ImageNet copy) and
`redis_cfg.redis_rendezvous_world_size` (1: one standalone Redis per node — each node
caches what it reads; -1: Ray-managed Redis Cluster across nodes) cover multi-node
layouts.

The same recipe under an external launcher (torchrun; `launcher=mp`):

```bash
torchrun --standalone --nproc-per-node=2 \
  -m tinyexp.examples.mae_exp launcher=mp
```

On a cluster, scale workers (Ray fills a placement group per worker: 1 GPU + 12 CPUs
by default) and adjust `accum_iter` so the effective batch stays 4096, exactly as
PRETRAIN.md describes:

```bash
python -m tinyexp.examples.mae_exp ray_cfg.ray_num_worker=64 accum_iter=1  # 8 nodes x 8 GPUs
```

Resume (rewrites `output/<exp_name>/last.ckpt` every epoch):

```bash
python -m tinyexp.examples.mae_exp ray_cfg.ray_num_worker=8 \
    resume_from=output/mae_exp/last.ckpt
```

Warm-start the autoencoder from another MAE checkpoint (official `--finetune` branch):

```bash
python -m tinyexp.examples.mae_exp ray_cfg.ray_num_worker=8 \
    module_cfg.pretrained_from=/path/to/mae_pretrain_vit_base.pth
```

Remote logging on a cluster run (tensorboard is not ported):

```bash
python -m tinyexp.examples.mae_exp ray_cfg.ray_num_worker=8 wandb_cfg.enable_wandb=true
```

Quick real-data training smoke (20 steps, two GPUs):

```bash
python -m tinyexp.examples.mae_exp ray_cfg.ray_num_worker=2 epochs=1 max_train_steps=20 accum_iter=1
```

Eval cross-check against the official finetuned checkpoint (downloaded once into
`~/.cache/torch/hub/checkpoints/`; batch-16 and 2-GPU variants as in the module
docstring):

```bash
python -m tinyexp.examples.mae_exp mode=eval \
    module_cfg.pretrained_from=https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth
```

## Linear probe (`mae_linprobe_exp`)

Source provenance: the same upstream commit pinned above, specifically
[`main_linprobe.py`](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_linprobe.py),
[`util/lars.py`](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lars.py),
[`util/crop.py`](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/crop.py),
the train/eval loops in `engine_finetune.py`, and the shared `util/lr_sched.py` schedule.
The reference recipe and 67.8% result come from upstream
[`FINETUNE.md` (linear probing)](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/FINETUNE.md).
Two-GPU Ray training/resume/eval smokes passed; **full 90-epoch ImageNet accuracy is
unvalidated**. No probe accuracy is claimed here.

`MaeLinprobeExp` subclasses `MaeExp`. It freezes the ViT-B encoder, takes the unmasked
CLS-token feature (not global pooling), and trains only a Linear classifier behind
`BatchNorm1d(affine=False, eps=1e-6)`. LARS uses the official bias/norm exclusions and no
weight decay. The train transform is the official linear-probe
`RandomResizedCrop(224, scale=(0.08, 1.0), bicubic) → HFlip → ToTensor → Normalize`; validation
uses the existing deterministic eval transform.

Recipe defaults: 90 epochs, 10 warmup epochs, `blr=0.1`, batch 512/GPU, and `accum_iter=4`.
The inherited Ray worker default is **one**. The primary commands below explicitly select
**eight GPUs, batch 2048/GPU, and `accum_iter=1`**, for large-memory cluster validation:

| Setting | Official recipe | Eight-GPU validation below |
| --- | --- | --- |
| GPUs | 32 (4 nodes × 8) | 8 |
| Physical batch/GPU | 512 | 2048 |
| Gradient accumulation | 1 | 1 (none) |
| Effective batch | 16384 | 16384 |
| Base LR / peak scaled LR | 0.1 / 6.4 | 0.1 / 6.4 |

The scaled peak is `lr = blr * eff_batch / 256 = 6.4`; warmup starts at zero.
BN is **ordinary local BatchNorm**, not SyncBN. The eight-GPU command matches the official
**effective batch and LR**, but BN sees **2048 rather than 512 samples** per forward,
so it is **not a strictly numerically equivalent reproduction** of the official recipe.
The official **67.8%** top-1 reference uses a **1600-epoch** pretraining checkpoint;
it is not a promised result for your **800-epoch** encoder.

Batch 2048 has not been memory-tested locally; the two-GPU smokes below used small batches.
If memory is insufficient, batch 512/GPU with `accum_iter=4` on eight GPUs also preserves
effective batch 16384 and uses the official per-forward BN batch size, though its BN running
statistics update four times per optimizer step. Batch 128 with accumulation 16 is another
lower-memory option with different BN statistics. Record the physical batch and accumulation
used. An incomplete accumulation window at the end of an epoch is discarded, as upstream does.

Initialize from your TinyExp pretraining checkpoint (eight GPUs, 2048/GPU, no accumulation):

```bash
export IMAGENET_HOME=/path/to/imagenet
python -m tinyexp.examples.mae_linprobe_exp ray_cfg.ray_num_worker=8 \
    dataloader_cfg.train_batch_size_per_device=2048 accum_iter=1 \
    module_cfg.pretrained_from=/mnt/jfs-zane-research/outputs/tinyexp/mae_exp/last.ckpt \
    output_root=/mnt/jfs-zane-research/outputs/tinyexp
```

`module_cfg.pretrained_from` accepts a TinyExp checkpoint containing `model_state_dict` or
an official MAE checkpoint containing `model`. This is encoder initialization for a **new
probe**, not pretraining-state resume. The commands here explicitly set shared storage:
`/mnt/jfs-zane-research/outputs/tinyexp/mae_linprobe_exp/`, separate from `mae_exp/`.
Without the `output_root` override, the default output is `./output/mae_linprobe_exp/`.

Every epoch evaluates the **full** ImageNet val set for top-1/top-5, appends per-epoch JSON
stats (`test_loss`, `test_acc1`, `test_acc5`, alongside training stats) to `log.txt`, and writes
`last.ckpt`; an improved val top-1 also writes `best.ckpt`.

Evaluate the trained probe or resume probe training using `resume_from` (a **probe**
checkpoint, not the source autoencoder checkpoint):

```bash
python -m tinyexp.examples.mae_linprobe_exp mode=eval ray_cfg.ray_num_worker=8 \
    resume_from=/mnt/jfs-zane-research/outputs/tinyexp/mae_linprobe_exp/best.ckpt \
    output_root=/mnt/jfs-zane-research/outputs/tinyexp

python -m tinyexp.examples.mae_linprobe_exp ray_cfg.ray_num_worker=8 \
    dataloader_cfg.train_batch_size_per_device=2048 accum_iter=1 \
    resume_from=/mnt/jfs-zane-research/outputs/tinyexp/mae_linprobe_exp/last.ckpt \
    output_root=/mnt/jfs-zane-research/outputs/tinyexp
```

Resume restores the probe training state and continues at `epoch + 1`; it does not need
`module_cfg.pretrained_from` again. Keep the same physical batch, accumulation, and worker
count on resume: these CLI overrides are not automatically restored from the checkpoint.
Independent eval uses its separate validation batch setting, not the training batch.

Inherited `mask_ratio`, `module_cfg.norm_pix_loss`, and `knn_cfg` do not participate in probe
training/evaluation. Reconstruction monitors are not used: full-val classification runs
every epoch. The pretraining k-NN monitor and its protocol remain unchanged; k-NN is still
not linear probing.

**Two-GPU Ray smoke record:** loading a cached official encoder on fake images passed.
A separate run loaded a full TinyExp MAE payload (`epoch=799`) on a real-ImageNet symlink
subset (1000 train + 1000 val images), with batch 16/GPU, `accum_iter=2`, and 20 microsteps.
Every encoder tensor stayed bit-identical; head weight/bias changed, BN tracked 20 batches,
and LARS momentum was saved. The probe checkpoint correctly started at epoch 0/step 20,
rather than inheriting pretraining epoch 799. Resume reached epoch 1/step 24, and a separate
`best.ckpt` evaluation exited successfully. The original pretraining checkpoint was left
untouched. The supplied `/mnt/jfs-zane-research/outputs/tinyexp/mae_exp/last.ckpt` was
inaccessible in the verification environment; these checks validate the payload format,
not that specific encoder or its downstream accuracy.

## Data preparation

ImageNet in the standard ImageFolder layout (`train/` for pretraining or linear probing,
`val/` for evaluation):

```
$IMAGENET_HOME/train/n01440764/xxx.JPEG
...
$IMAGENET_HOME/val/n01440764/xxx.JPEG
```

## Cross-check record

Checkpoints: finetuned ViT-B
`https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth` (FINETUNE.md
md5 prefix `1b25e9`, 346,326,087 bytes).

**L1 — numerical (CI, `tests/examples/test_mae_exp.py`):**

- Classifier: ViT-B parameter count 86,567,656 ("86.57M" in the official log) and the
  official checkpoint key layout (`fc_norm.*`, no `norm.*`, `pos_embed` `(1, 197, 768)`,
  152 keys); the composed MAE forward is bit-identical to timm's native stages for both
  `global_pool=True/False` (atol=0, rtol=0).
- Autoencoder: ViT-B parameter count 111,907,840; masking removes exactly
  `int(L * mask_ratio)` patches per sample; `patchify`/`unpatchify` are exact inverses;
  the `norm_pix_loss` value matches a manual recomputation.
- Loader: round-trips exactly for both model families (`--resume`/`--finetune`
  warm-start), drops + re-initializes a mismatched head; `interpolate_pos_embed`
  resizes a 16×16-grid embedding into a 4×4-grid model and is a no-op (same tensor
  object) when grids match.
- Schedule: linear warmup → cosine values at the boundaries, `lr_scale` honored.
- Optimizer: `blr * (batch * accum * world) / 256` linear rule, betas (0.9, 0.95),
  weight-decay grouping (bias/norm at 0, frozen pos embeds excluded).
- Transforms: eval `Resize(256, bicubic) → CenterCrop(224) → ToTensor → Normalize` with
  the 384-input `crop_pct 1.0` branch; train `RandomResizedCrop(224, scale=(0.2, 1.0),
  bicubic) → HFlip → ToTensor → Normalize`.
- Run path: CPU fake-data training smoke (checkpoint payload + per-epoch JSON stats) and
  a resume run that continues at `epoch + 1`.

**L2 — end-to-end eval accuracy (single RTX 4080, AMP autocast, batch 64, sequential
val):**

| | acc@1 | acc@5 | loss |
| --- | --- | --- | --- |
| official (FINETUNE.md, V100) | 83.664 | 96.530 | 0.731 |
| this port (RTX 4080) | 83.746 | 96.540 | 0.731 |

The loss matches to three decimals; acc@1 is +0.082 (41 of 50000 images) and acc@5
+0.010. The pipeline is byte-identical (same transform, sampler, and autocast eval), so
the residual is fp16-autocast matmul reduction-order variance across GPU architectures —
the same class of delta recorded for the DeiT-S port (79.82 vs 79.8 official). Eval wall
time 14:25, dataloader-bound.

**L3 — pretraining 跑通 (2× RTX 4080, DDP, real ImageNet train):**

- `ray_cfg.ray_num_worker=2 epochs=1 max_train_steps=20` (recorded before accum_iter defaulted to 8; ran with accum_iter=1): 20 steps × 128 images, loss
  1.847 → 1.846 (init-level, `norm_pix_loss` scale), 4.3 GB peak GPU memory per rank,
  `actual lr 7.50e-05` (= blr 1.5e-4 × 128/256), per-iteration warmup visible in the lr
  meter, `last.ckpt` (1.2 GB: model + optimizer + scaler) and per-epoch JSON stats
  written.
- `resume_from=output/mae_exp/last.ckpt epochs=2 max_train_steps=30`: continued at
  `Epoch: [1]`, loss 1.846 → 1.734 (optimizer/scaler states restored, training
  progresses), lr resumed at the correct fractional-epoch warmup value (1.88e-6),
  stats appended.
- Same smoke with the Redis cache on (default): six shards auto-started on ports
  7000-7005 by the Ray driver and stopped with the run; the cached run's loss matches
  the uncached run digit-for-digit (1.8465/1.8458 — bit-identical samples, D11).
- Monitors on the same smoke (`eval_every_n_epochs=1 knn_cfg.knn_every_n_epochs=1`):
  `{"train_loss": 1.8458, "val_loss": 1.8460, "knn_top1": 0.78, "epoch": 0}` — the
  k-NN top-1 is the expected random-initialization baseline (chance 0.1% on 1000
  classes) after 10 training steps; it is the curve that should climb as pretraining
  progresses. Recon monitor: 56 s per full-val pass; the first k-NN bank extraction
  paid the cold Redis cache (~7.5 min on a slow external disk, 51200 images) and
  subsequent evaluations read the warm bytes.

Full-recipe runs (800 epochs, effective batch 4096) are deferred to a GPU cluster;
record the cluster numbers here when available.

## Not ported (this round)

Finetuning training (`main_finetune.py` training loop),
`mae_vit_large_patch16`/`mae_vit_huge_patch14`/`vit_large_patch16`/`vit_huge_patch14`,
layer-wise lr decay, submitit, and tensorboard. Linear probing is smoke-verified as documented
above; full 90-epoch ImageNet accuracy remains unvalidated.
