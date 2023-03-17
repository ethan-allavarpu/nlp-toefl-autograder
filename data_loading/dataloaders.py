from typing import Tuple
import torch
from torch.utils.data import DataLoader
import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
import warnings
import numpy as np

def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1.1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

def get_data_loaders(dataset: torch.utils.data.Dataset, val_size: float = 0.0, test_size: float=0.0,
batch_size: int = 32, val_batch_size: int = 16, test_batch_size: int = 16, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if val_size==0.0 and test_size==0.0:
        train_dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
        return train_dl
    # split the dataset into train, val, and test
    datasets = random_split(dataset, [1-val_size-test_size, val_size, test_size], generator=torch.Generator().manual_seed(1))
    
    train_dl = DataLoader(datasets[0], batch_size=batch_size, shuffle=True,num_workers=num_workers, )
    val_dl = DataLoader(datasets[1], batch_size=val_batch_size, shuffle=False,num_workers=num_workers)
    test_dl = DataLoader(datasets[2], batch_size=test_batch_size, shuffle=False,num_workers=num_workers)
    
    return train_dl, (val_dl if len(val_dl)>0 else None), (test_dl if len(test_dl)>0 else None)

def split_on_indices(dataset: torch.utils.data.Dataset, index_col: str, val_size: float = 0.0, test_size: float=0.0,
batch_size: int = 32, val_batch_size: int = 16, test_batch_size: int = 16, num_workers: int = 0, seed: int = 0) -> DataLoader:
    idx = np.array(list(set(dataset.data[index_col])))
    np.random.seed(1+seed)
    np.random.shuffle(idx)
    l = idx.shape[0]
    test_idx = idx[: int(l * test_size)]
    val_idx = idx[int(l * test_size) : int(l * (val_size+ test_size))]
    train_idx = idx[int(l * (val_size+test_size)):] 
    train_ds = Subset(dataset, np.where(np.isin(dataset.data[index_col], train_idx))[0])
    val_ds = Subset(dataset, np.where(np.isin(dataset.data[index_col], val_idx))[0])
    test_ds = Subset(dataset, np.where(np.isin(dataset.data[index_col], test_idx))[0])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,num_workers=num_workers, )
    val_dl = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False,num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False,num_workers=num_workers)

    return train_dl, val_dl, test_dl