"""``HoldOnesampleDataLoader(dataloader)``: yields a deep copy of the first batch forever.

Use it to profile or debug the model without data-loading cost; ``len()`` reports the wrapped length.
"""

import copy

__all__ = ["HoldOnesampleDataLoader"]


class HoldOnesampleDataLoader:
    """
    A fake dataloader that holds the first batch of the original dataloader and returns a deep copy of it
    on every ``next()``; iteration never stops, so bound the loop yourself (``len()`` is the wrapped length).
    This is useful for testing and profiling purposes, where we want to test the model with a single in-memory sample.
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        for data in self.dataloader:
            self.sample = data
            break

    def __iter__(self):
        return self

    def __next__(self):
        return copy.deepcopy(self.sample)

    def __len__(self):
        return len(self.dataloader)
