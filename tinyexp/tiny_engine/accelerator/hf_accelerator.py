try:
    from accelerate import Accelerator

    class HFAccelerator(Accelerator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._set_attributes()

        def _set_attributes(self):
            for attr_name in ("rank", "world_size", "local_rank"):
                if hasattr(self, attr_name):
                    raise AttributeError(attr_name)
            self.rank = self.process_index
            self.world_size = self.num_processes
            self.local_rank = self.local_process_index

except ImportError:
    import warnings

    warnings.warn("accelerate is not installed, please install it with `pip install accelerate`", stacklevel=2)
    # HFAccelerator will not be defined if accelerate is not installed
