try:
    import contextlib

    from accelerate import Accelerator

    class HFAccelerator(Accelerator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._destroyed = False
            self._set_attributes()

        def _set_attributes(self):
            """Alias accelerate's indices under the names the other accelerators use.

            Raises when this accelerate version already defines one of the names,
            so a future conflict fails loudly instead of being silently shadowed.
            """
            for attr_name in ("rank", "world_size", "local_rank"):
                if hasattr(self, attr_name):
                    raise AttributeError(  # noqa: TRY003
                        f"accelerate's Accelerator already defines {attr_name!r}, which "
                        "HFAccelerator aliases onto process_index/num_processes/"
                        "local_process_index; refusing to silently shadow it. Check the "
                        "installed accelerate version."
                    )
            self.rank = self.process_index
            self.world_size = self.num_processes
            self.local_rank = self.local_process_index

        def optimizer_step(self, optimizer):
            # A prepared optimizer is an AcceleratedOptimizer whose own step()
            # applies accelerate's loss scaling; autocast() and prepare_model()
            # (with its device_placement signature) are inherited unchanged.
            return optimizer.step()

        def destroy(self) -> None:
            if self._destroyed:
                return
            self.end_training()
            self._destroyed = True

        def __del__(self) -> None:
            with contextlib.suppress(Exception):
                self.destroy()

except ImportError:
    import warnings

    warnings.warn("accelerate is not installed, please install it with `pip install accelerate`", stacklevel=2)
    # HFAccelerator will not be defined if accelerate is not installed
