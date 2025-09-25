from accelerate import Accelerator


class HFAccelerator(Accelerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_attributes()

    def _set_attributes(self):
        assert not hasattr(self, "rank"), "rank already exists"
        assert not hasattr(self, "world_size"), "world_size already exists"
        assert not hasattr(self, "local_rank"), "local_rank already exists"
        self.rank = self.process_index
        self.world_size = self.num_processes
        self.local_rank = self.local_process_index
