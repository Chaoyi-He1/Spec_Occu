import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from misc import get_rank, get_world_size
from typing import Iterator, Optional, List, TypeVar
from torch.utils.data.sampler import Sampler
import math


__all__ = ["DistributedSampler", ]
T_co = TypeVar('T_co', covariant=True)


class Custom_DistributedSampler(DistributedSampler):
    """
    Customized DistributedSampler for the DataLoader.
    Mostly copied from torch.utils.data.distributed.DistributedSampler
    Just change the __iter__ function to repeat the dataset for each epoch multiple times.
    """
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if drop_last and len(super().dataset) % super().num_replicas != 0:  
            # type: ignore[arg-type]

            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples_extend = math.ceil(
                (len(super().dataset) - super().num_replicas) / super().num_replicas  
                # type: ignore[arg-type]
            )
        else:
            self.num_samples_extend = math.ceil(
                len(super().dataset) / super().num_replicas)  
            # type: ignore[arg-type]
        self.total_size_extend = self.num_samples_extend * super().num_replicas
    
    def __iter__(self) -> Iterator[T_co]:
        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(super().seed + super().epoch)

