import pytest
import ray
import torch

from tinyexp.tiny_engine.accelerator import DDPAccelerator
from tinyexp.utils.ray_utils import get_num_worker_options, get_placement_group


@ray.remote
class DDPAcceleratorProxy:
    def __init__(self):
        # This will initialize the process group using env vars from get_num_gpus_worker_options
        self.accelerator = DDPAccelerator()

    def test_reduce_sum(self):
        # 1. Create tensor on the correct device for this worker.
        device = self.accelerator.device
        # Each worker creates a tensor with its rank [0], [1], etc.
        tensor_to_sum = torch.tensor([self.accelerator.rank], device=device, dtype=torch.float32)

        # 2. Call reduce_sum. This will sum the tensors from all workers.
        # For 2 workers, the ranks are 0 and 1. The sum is 0 + 1 = 1.
        res = self.accelerator.reduce_sum(tensor_to_sum)

        # 3. The result on all workers should be the sum of all ranks.
        # Sum of 0 to n-1 is n * (n-1) / 2
        world_size = self.accelerator.world_size
        expected_val = (world_size * (world_size - 1)) / 2
        expected_result = torch.tensor([expected_val], device=device, dtype=torch.float32)

        assert torch.equal(res, expected_result)
        return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
class TestDDPAcceleratorWithRay:
    def test_ddp_accelerator(self, ray_session):
        num_workers = 2
        # This utility correctly sets up env vars and placement groups.
        pg = get_placement_group(
            num_worker=num_workers,
            num_gpus_per_worker=1.0,  # Each worker gets 1 GPU
            num_cpus_per_worker=4,  # Each worker gets 1 CPU
        )
        gpu_per_actor = 0.2

        options_list1 = get_num_worker_options(pg, num_workers, gpu_ratio=gpu_per_actor)
        worker_group1 = [DDPAcceleratorProxy.options(**options).remote() for options in options_list1]
        run_futures1 = [worker.test_reduce_sum.remote() for worker in worker_group1]

        options_list2 = get_num_worker_options(pg, num_workers, gpu_ratio=gpu_per_actor)
        worker_group2 = [DDPAcceleratorProxy.options(**options).remote() for options in options_list2]
        run_futures2 = [worker.test_reduce_sum.remote() for worker in worker_group2]

        results1 = ray.get(run_futures1)
        results2 = ray.get(run_futures2)
        assert all(results1)
        assert all(results2)
