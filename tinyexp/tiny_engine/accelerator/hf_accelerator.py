from accelerate import Accelerator


class HFAccelerator(Accelerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_attributes()

    def _set_attributes(self):
        self.rank = self.process_index
        self.world_size = self.num_processes
        self.local_rank = self.local_process_index
