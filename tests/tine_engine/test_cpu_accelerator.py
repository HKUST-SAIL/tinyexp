import ray
import torch

from tinyexp.tiny_engine.accelerator import CPUAccelerator
from tinyexp.utils.ray_utils import get_num_worker_options, get_placement_group


@ray.remote
class CPUAcceleratorProxy:
    def __init__(self):
        self.accelerator = CPUAccelerator()

    def test_reduce_sum(self):
        device = self.accelerator.device
        tensor_to_sum = torch.tensor([self.accelerator.rank], device=device, dtype=torch.float32)
        res = self.accelerator.reduce_sum(tensor_to_sum)
        world_size = self.accelerator.world_size
        expected_val = (world_size * (world_size - 1)) / 2
        expected_result = torch.tensor([expected_val], device=device, dtype=torch.float32)
        assert torch.equal(res, expected_result)
        return True


class TestCPUAcceleratorWithRay:
    def test_ddp_accelerator(self, ray_session):
        num_worker = 2
        pg = None
        try:
            # This utility correctly sets up env vars and placement groups.
            pg = get_placement_group(
                num_worker=num_worker,
                num_gpus_per_worker=0.0,  # CPU workers, so no GPUs
                num_cpus_per_worker=2,  # Each worker gets 2 CPUs
            )
            options_list = get_num_worker_options(pg, num_worker=num_worker, gpu_ratio=0.0)
            # Create the remote actors.
            worker_group = [CPUAcceleratorProxy.options(**options).remote() for options in options_list]

            # Run the test method on all workers and wait for them to complete.
            run_futures = [worker.test_reduce_sum.remote() for worker in worker_group]
            results = ray.get(run_futures)

            # Verify that all tests passed.
            assert all(results)
        finally:
            if pg:
                ray.util.remove_placement_group(pg)
