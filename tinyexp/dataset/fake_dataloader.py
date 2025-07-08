import copy

__all__ = ["HoldOnesampleDataLoader"]


class HoldOnesampleDataLoader:
    """
    A fake dataloader that holds one sample from the original dataloader.
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
