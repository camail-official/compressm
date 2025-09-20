import torch
from torch.utils.data import DataLoader

class PyTorchDataloaderWrapper(DataLoader):
    """
    A wrapper class that adapts a PyTorch Dataset to mimic the interface of the custom Dataloader class.
    It provides .loop() and .loop_epoch() methods for compatibility, with batch_size specified at initialization.
    No key is required for .loop(), and shuffling is handled internally without a specific seed.

    Args:
        dataset (torch.utils.data.Dataset): The PyTorch dataset to wrap.
        batch_size (int): The batch size to use for batching.
        **kwargs: Additional keyword arguments to pass to DataLoader (e.g., num_workers, pin_memory, etc.).
                  Note: Do not pass batch_size, shuffle, or drop_last here, as they are handled internally.
    """
    def __init__(self, **kwargs):
        super().__init__(num_workers=0, pin_memory=True, **kwargs)
        self.kwargs = kwargs

        self.size = len(self.dataset)

    def loop(self, *args, **kwargs):
        if self.kwargs["batch_size"] == self.kwargs["dataset"].__len__():
            whole_batch = next(super().__iter__())
            while True:
                yield whole_batch
        else:
            # For smaller batch sizes, continuously iterate through shuffled epochs
            while True:
                # Each call to super().__iter__() creates a new iterator with fresh shuffling
                # (if shuffle=True was passed during initialization)
                for batch in super().__iter__():
                    yield batch

    def loop_epoch(self, *args, **kwargs):
        # One epoch without shuffling, including last incomplete batch if any
        for batch in super().__iter__():
            yield batch