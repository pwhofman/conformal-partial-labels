from typing import Tuple
from torch import Tensor
from torch.utils.data import Dataset

class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    """Dataset wrapping tensors.
    Each sample will be retrieved by indexing tensors along the first dimension.
    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return (*tuple(tensor[index] for tensor in self.tensors), index)

    def __len__(self):
        return self.tensors[0].size(0)