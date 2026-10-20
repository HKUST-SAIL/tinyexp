from contextlib import suppress

import pytest
import ray
import torch

from tinyexp.tiny_engine.accelerator import DDPAccelerator
from tinyexp.utils.ray_utils import (
    get_num_worker_options,
    get_placement_group,
    get_placement_group_node_ids,
    start_ray_rendezvous_store,
)


@ray.remote
class DDPAcceleratorProxy:
    def __init__(self):
        # Initializes the process group from the env vars set by get_num_worker_options.
        self.accelerator = DDPAccelerator()

    def test_reduce(self):
        device = self.accelerator.device
        world_size = self.accelerator.world_size
        expected_val = (world_size * (world_size - 1)) / 2  # sum of ranks 0..n-1

        res = self.accelerator.reduce(torch.tensor([self.accelerator.rank], device=device, dtype=torch.float32))
        assert torch.equal(res, torch.tensor([expected_val], device=device, dtype=torch.float32))

        # nccl cannot reduce cpu tensors directly; reduce must transparently
        # stage them on the device and return the result on the input device
        # (regression: metric sync after an epoch crashed with "No backend type
        # associated with device type cpu" under DDP+nccl).
        cpu_tensor = torch.tensor([float(self.accelerator.rank)], dtype=torch.float64)
        cpu_res = self.accelerator.reduce(cpu_tensor, reduction="sum")
        assert cpu_res.device.type == "cpu"
        assert torch.equal(cpu_res, torch.tensor([expected_val], dtype=torch.float64))
        # The caller's tensor is never mutated in place.
        assert torch.equal(cpu_tensor, torch.tensor([float(self.accelerator.rank)], dtype=torch.float64))
        return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_two_ddp_groups_share_one_placement_group(ray_session):
    num_workers = 2
    pg = get_placement_group(num_worker=num_workers, num_gpus_per_worker=1.0, num_cpus_per_worker=4)
    node_ids = get_placement_group_node_ids(pg, num_workers)

    worker_group = []
    rendezvous_actors = []
    try:
        # Two independent process groups, each with its own rendezvous store, on fractional GPUs.
        for _ in range(2):
            rendezvous_actor, master_addr, master_port = start_ray_rendezvous_store(pg, num_workers, timeout_s=30)
            rendezvous_actors.append(rendezvous_actor)
            options_list = get_num_worker_options(
                pg,
                num_workers,
                gpu_ratio=0.2,
                num_cpus_per_worker=1,
                master_addr=master_addr,
                master_port=master_port,
                node_ids=node_ids,
            )
            worker_group += [DDPAcceleratorProxy.options(**options).remote() for options in options_list]

        assert all(ray.get([worker.test_reduce.remote() for worker in worker_group], timeout=60))
    finally:
        for actor in worker_group + rendezvous_actors:
            with suppress(Exception):
                ray.kill(actor, no_restart=True)
        ray.util.remove_placement_group(pg)
