"""
Estimate pi with distributed Monte Carlo sampling, launched by Ray in ``mode="run"``.

This example shows that tinyexp can drive general distributed programs (no deep
learning, no dataloader): each Ray worker samples random points in the unit square,
counts how many fall inside the unit circle (x*x + y*y <= 1.0), and the counts are
all-reduced to estimate pi = 4 * inside / total.

Usage::

    python -m tinyexp.examples.pi_exp
    python -m tinyexp.examples.pi_exp pi_cfg.total_samples=100000000 ray_cfg.ray_num_worker=4
"""

from dataclasses import dataclass, field

import torch

from tinyexp import TinyExp, store_and_run_exp
from tinyexp.exp_mixins import LoggerCfgMixin, RayCfgMixin


@dataclass(repr=False)
class Exp(TinyExp, RayCfgMixin, LoggerCfgMixin):
    @dataclass
    class RayCfg(RayCfgMixin.RayCfg):
        ray_num_worker: int = 2
        ray_num_cpus_per_worker: int = 1
        ray_num_gpus_per_worker: float = 0.0  # 0.0 means do not use GPU

    ray_cfg: RayCfg = field(default_factory=RayCfg)
    mode: str = "run"
    launcher: str = "ray"

    @dataclass
    class PiCfg:
        # Total number of Monte Carlo samples across all workers.
        total_samples: int = 10_000_000
        # Number of samples drawn per iteration on each worker, to bound memory usage.
        chunk_size: int = 1_000_000
        # Base random seed; each worker offsets it by its rank.
        seed: int = 42

    pi_cfg: PiCfg = field(default_factory=PiCfg)

    def run(self) -> None:
        if self.mode != "run":
            raise NotImplementedError(f"Mode {self.mode} is not implemented")

        from tinyexp.tiny_engine.accelerator import CPUAccelerator

        accelerator = CPUAccelerator()
        logger = self.logger_cfg.build_logger(save_dir=self.get_run_dir(), distributed_rank=accelerator.rank)
        self.print_cfg(logger)

        try:
            pi = self._estimate_pi(accelerator)
            if accelerator.is_main_process:
                logger.info(f"pi ~= {pi:.6f} (error={abs(pi - torch.pi):.6f}, samples={self.pi_cfg.total_samples})")
        finally:
            accelerator.destroy()

    def _estimate_pi(self, accelerator) -> float:
        generator = torch.Generator().manual_seed(self.pi_cfg.seed + accelerator.rank)
        samples_per_rank = self.pi_cfg.total_samples // accelerator.world_size

        inside = 0
        remaining = samples_per_rank
        while remaining > 0:
            n = min(remaining, self.pi_cfg.chunk_size)
            points = torch.rand(n, 2, generator=generator)
            inside += int((points.square().sum(dim=1) <= 1.0).sum())
            remaining -= n

        total_inside = accelerator.reduce_sum(torch.tensor(inside, dtype=torch.float64))
        return 4.0 * total_inside.item() / (samples_per_rank * accelerator.world_size)


if __name__ == "__main__":
    store_and_run_exp(Exp)
