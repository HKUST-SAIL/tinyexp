"""Distributed Monte Carlo estimate of pi: a non-deep-learning workload driven by TinyExp.

Each worker samples points in the unit square, counts those inside the unit circle, and the counts are
all-reduced to estimate pi = 4 * inside / total. No dataloader, no model; also the fastest smoke test.

Modes : run | help
Run   :
    python -m tinyexp.examples.pi_exp mode=help
    python -m tinyexp.examples.pi_exp                     # default: 2 Ray CPU workers
    python -m tinyexp.examples.pi_exp pi_cfg.total_samples=100000000 ray_cfg.ray_num_worker=4
    python -m tinyexp.examples.pi_exp launcher=mp pi_cfg.total_samples=100000
    accelerate launch --cpu --num-processes=1 -m tinyexp.examples.pi_exp launcher=mp
Env   : none
Needs : tinyexp[pytorch]
Out   : ``pi ~= ...`` on stdout (by rank 0, or by the Ray driver); log in <output_root>/pi_exp/log.txt
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

from tinyexp import TinyExp, store_and_run_exp
from tinyexp.exceptions import UnknownExperimentModeError
from tinyexp.exp_mixins import LoggerCfgMixin, RayCfgMixin
from tinyexp.tiny_engine.accelerator import AcceleratorProtocol


@dataclass(repr=False)
class Exp(TinyExp, RayCfgMixin, LoggerCfgMixin):
    mode: str = "run"
    launcher: str = "ray"

    @dataclass
    class RayCfg(RayCfgMixin.RayCfg):
        ray_num_worker: int = 2
        ray_num_cpus_per_worker: int = 1
        ray_num_gpus_per_worker: float = 0.0  # 0.0 means do not use GPU

    ray_cfg: RayCfg = field(default_factory=RayCfg)

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
            raise UnknownExperimentModeError(self.mode, allowed=("run", "help"))

        self._run_result: Optional[str] = None
        from tinyexp.tiny_engine.accelerator import CPUAccelerator

        accelerator = CPUAccelerator()
        try:
            self._run(accelerator)
        finally:
            accelerator.destroy()

    def _run(self, accelerator: AcceleratorProtocol) -> None:
        logger = self.logger_cfg.build_logger(save_dir=self.get_run_dir(), distributed_rank=accelerator.rank)
        self.print_cfg(logger)

        pi = self._estimate_pi(accelerator)
        if accelerator.is_main_process:
            result = f"pi ~= {pi:.6f} (error={abs(pi - torch.pi):.6f}, samples={self.pi_cfg.total_samples})"
            self._run_result = result
            if self.launcher != "ray":
                print(result, flush=True)

    def get_ray_run_result(self) -> Optional[str]:
        return self._run_result

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

        total_inside = accelerator.reduce(torch.tensor(inside, dtype=torch.float64), reduction="sum")
        return 4.0 * total_inside.item() / (samples_per_rank * accelerator.world_size)


if __name__ == "__main__":
    store_and_run_exp(Exp)
