import pytest
import ray
import torch

from tinyexp.tiny_engine.accelerator import DDPAccelerator
from tinyexp.utils.ray_utils import get_num_gpus_worker_options


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
    @pytest.fixture(autouse=True)
    def setup_ray(self):
        # Use a fixture to ensure Ray is initialized and shut down cleanly.
        # Exclude files to prevent `ray.init` from hanging on large directories.
        # This tells Ray to only package necessary python files.

        # print("Setting up Ray for testing...")
        runtime_env = {
            "working_dir": ".",
            "excludes": ["*.md", "data/", "tests/", ".git/", ".venv/", "output/", "outputs/", "site/"],
        }
        if not ray.is_initialized():
            ray.init(runtime_env=runtime_env)
        # print("Ray initialized for testing.")

        yield  # This is where the test runs

        # print("Shutting down Ray...")
        ray.shutdown()
        # print("Ray shut down.")

    def test_ddp_accelerator(self):
        num_workers = 2
        # This utility correctly sets up env vars and placement groups.
        options_list = get_num_gpus_worker_options(num_workers, num_cpus_per_gpu=1)

        # Create the remote actors.
        worker_group = [DDPAcceleratorProxy.options(**options).remote() for options in options_list]

        # Run the test method on all workers and wait for them to complete.
        run_futures = [worker.test_reduce_sum.remote() for worker in worker_group]
        results = ray.get(run_futures)

        # Verify that all tests passed.
        assert all(results)
