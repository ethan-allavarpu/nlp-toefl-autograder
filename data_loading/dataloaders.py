from typing import Tuple
import torch
from torch.utils.data import DataLoader



def get_data_loaders(dataset: torch.utils.data.Dataset, val_size: float = 0.0, test_size: float=0.0,
batch_size: int = 32, val_batch_size: int = 16, test_batch_size: int = 16, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if val_size==0.0 and test_size==0.0:
        train_dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
        return train_dl
    # split the dataset into train, val, and test
    datasets = torch.utils.data.random_split(dataset, [1-val_size-test_size, val_size, test_size], generator=torch.Generator().manual_seed(1))
    
    train_dl = DataLoader(datasets[0], batch_size=batch_size, shuffle=True,num_workers=num_workers, )
    val_dl = DataLoader(datasets[1], batch_size=val_batch_size, shuffle=False,num_workers=num_workers)
    test_dl = DataLoader(datasets[2], batch_size=test_batch_size, shuffle=False,num_workers=num_workers)
    
    return train_dl, (val_dl if len(val_dl)>0 else None), (test_dl if len(test_dl)>0 else None)

