import ray
import torch
from torch import nn

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

    def test_model_parameters_are_synchronized(self):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(self.accelerator.rank + 1)
        prepared_model = self.accelerator.prepare_model(model)
        return prepared_model.module.weight.detach().item()


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

            model_results = ray.get([worker.test_model_parameters_are_synchronized.remote() for worker in worker_group])
            assert model_results == [1.0] * num_worker
        finally:
            if pg:
                ray.util.remove_placement_group(pg)
