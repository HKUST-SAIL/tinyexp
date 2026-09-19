"""SAIL-Recon inference: structured configuration, atomic stages, no training."""

# ruff: noqa: TRY003 -- keep actionable input errors local to this experiment.

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tinyexp import TinyExp, store_and_run_exp
from tinyexp.exp_mixins import LoggerCfgMixin
from tinyexp.sail_recon.io import cpu_predictions, image_paths, save_pointcloud, save_poses, sha256_file, uniform_sample
from tinyexp.sail_recon.models.model import SailRecon
from tinyexp.sail_recon.trace import Trace, tensor_record
from tinyexp.sail_recon.utils.load_fn import load_and_preprocess_images
from tinyexp.tiny_engine.accelerator import AcceleratorProtocol, DDPAccelerator


@dataclass(repr=False)
class Exp(TinyExp, LoggerCfgMixin):
    mode: str = "run"
    launcher: str = "mp"
    exp_name: str = "sail_recon"

    @dataclass
    class DataCfg:
        image_dir: str = ""
        limit: int = 0
        stride: int = 1
        preprocess: str = "crop"

    @dataclass
    class ModelCfg:
        checkpoint: str = ""
        checkpoint_sha256: str = ""
        anchors: int = 100
        fix_rank: int = 300

    @dataclass
    class RuntimeCfg:
        seed: int = 42
        precision: str = "bfloat16"
        chunk_size: int = 20
        trace: bool = False
        require_a6000: bool = False
        release_global_blocks: bool = True

    data_cfg: DataCfg = field(default_factory=DataCfg)
    model_cfg: ModelCfg = field(default_factory=ModelCfg)
    runtime_cfg: RuntimeCfg = field(default_factory=RuntimeCfg)

    def run(self):
        if self.mode != "run" or self.launcher != "mp":
            raise ValueError("Use mode=run launcher=mp for single-GPU scene inference")
        accelerator = DDPAccelerator()
        try:
            self._run(accelerator)
        finally:
            accelerator.destroy()

    def _validate(self, accelerator: AcceleratorProtocol):
        if accelerator.world_size != 1:
            raise ValueError("One process/GPU owns a complete scene; multi-process inference is not supported")
        cfg = self.runtime_cfg
        if cfg.chunk_size <= 0 or self.model_cfg.anchors <= 0:
            raise ValueError("chunk_size and anchors must be positive")
        if cfg.precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("precision must be float32, float16, or bfloat16")
        gpu_name = torch.cuda.get_device_name(accelerator.device)
        if cfg.require_a6000 and "A6000" not in gpu_name:
            raise RuntimeError(f"A6000 required, found {gpu_name}")
        if cfg.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support bfloat16")
        paths = image_paths(self.data_cfg.image_dir, self.data_cfg.limit, self.data_cfg.stride)
        checkpoint = Path(self.model_cfg.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Set model_cfg.checkpoint to the NFS sailrecon.pt file: {checkpoint}")
        digest = sha256_file(checkpoint)
        if self.model_cfg.checkpoint_sha256 and digest != self.model_cfg.checkpoint_sha256:
            raise ValueError("Checkpoint SHA256 mismatch")
        return gpu_name, paths, checkpoint, digest

    def _run(self, accelerator: AcceleratorProtocol):
        gpu_name, paths, checkpoint, digest = self._validate(accelerator)
        cfg = self.runtime_cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        output = Path(self.get_run_dir())
        output.mkdir(parents=True, exist_ok=True)
        if (output / "predictions.pt").exists():
            raise FileExistsError(f"Refusing to overwrite existing predictions in {output}")
        logger = self.logger_cfg.build_logger(str(output), accelerator.rank)
        configuration = self.print_cfg(logger)
        (output / "config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
        model = SailRecon(kv_cache=True).eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model = accelerator.prepare(model)
        images = load_and_preprocess_images([str(p) for p in paths], mode=self.data_cfg.preprocess)
        indices = uniform_sample(len(images), min(len(images), self.model_cfg.anchors))
        trace = Trace(model) if cfg.trace else None
        manifest = {
            "checkpoint_sha256": digest,
            "images": [{"name": p.name, "sha256": sha256_file(p)} for p in paths],
            "preprocessed": tensor_record(images),
            "anchors": indices,
            "gpu": gpu_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "seed": cfg.seed,
            "precision": cfg.precision,
            "chunk_size": cfg.chunk_size,
            "fix_rank": self.model_cfg.fix_rank,
        }
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        predictions = []
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=getattr(torch, cfg.precision), enabled=cfg.precision != "float32"),
        ):
            model.build_scene(images[indices].to(accelerator.device), self.model_cfg.fix_rank)
            if trace:
                trace.cache(model)
            if cfg.release_global_blocks:
                model.release_global_blocks()
            for i, chunk in enumerate(images.split(cfg.chunk_size)):
                if trace:
                    trace.phase = f"query/{i}"
                predictions.extend(
                    cpu_predictions(model.localize(chunk.to(accelerator.device), self.model_cfg.fix_rank))
                )
        torch.cuda.synchronize()
        manifest["inference_seconds"] = time.perf_counter() - start
        manifest["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        if trace:
            trace.close()
            (output / "trace.json").write_text(json.dumps(trace.records, indent=2), encoding="utf-8")
        torch.save(predictions, output / "predictions.pt")
        # Export seed is independent of model initialization and cache sampling.
        torch.manual_seed(cfg.seed)
        save_pointcloud(predictions, output / "pred.ply")
        save_poses(predictions, output / "pred.txt")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"Saved {len(predictions)} views to {output}")


if __name__ == "__main__":
    store_and_run_exp(Exp)
