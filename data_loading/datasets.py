import pandas as pd
import torch
from typing import Sequence, Any, Dict, Tuple, TypeVar
import numpy as np
from datasets import load_dataset, Dataset

T_co = TypeVar('T_co', covariant=True)

class DefaultDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, input_col: str, target_cols: Sequence[str], index_col: str = None,
                 tokenizer: Any = None, tokenizer_params: Dict = None):
        self.data = pd.read_csv(file_path)
        # set index col so we can use it as a key
        if index_col:
            self.data.set_index(index_col, inplace=True)
        self.indices = self.data.index.tolist()

        self.inputs = pd.DataFrame(self.data[input_col])
        self.targets = pd.DataFrame(self.data[target_cols])
        # normalize the targets
        self.normalize_targets()
        # here we can provide the tokenizer!
        self.tokenizer = tokenizer
        # default tokenizer params
        self.tokenizer_params = tokenizer_params if tokenizer_params else {'padding': 
        'max_length', 
         'max_length': 512,
        # 'max_length': max([len(t) for t in self.tokenizer(self.inputs['full_text'].to_list())['input_ids']]),
         'truncation': True, 'return_tensors': 'pt'} 
    
    def normalize_targets(self, normalize_score: float = 100.0) -> None:
        self.targets = (self.targets) / self.targets.max(axis=0) * normalize_score

    def __getitem__(self, index: Any) -> T_co:
        idx = self.indices[index]
        if self.tokenizer:
            features = self.tokenizer(self.inputs.loc[idx].item(), **self.tokenizer_params)
        else:
            # input is already tokenized
            features = self.inputs.loc[idx].values
        return features, self.targets.loc[idx].values

    def __len__(self) -> int:
        return len(self.data)


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, path_name: str, input_col: str, target_cols: Sequence[str], 
                 tokenizer: Any = None, tokenizer_params: Dict = None):
        self.data = load_dataset(path_name, split='train')
        self.tokenizer = tokenizer

        tokenizer_params = tokenizer_params if tokenizer_params else {'sampling_rate':tokenizer.sampling_rate, 'max_length': tokenizer.sampling_rate, 
        'truncation': True}
        self.inputs = tokenizer(
        [x[input_col]['array'] for x in self.data], ** tokenizer_params)['input_values']

        self.targets = pd.DataFrame([[x[t] for t in target_cols] for x in self.data ], columns=target_cols)
        # normalize the targets
        self.normalize_targets()
    
    def normalize_targets(self, normalize_score: float = 100.0) -> None:
        self.targets = (self.targets) / self.targets.max(axis=0) * normalize_score

    def __getitem__(self, index: Any) -> T_co:
        return self.inputs[index], self.targets.loc[index].values

    def __len__(self) -> int:
        return len(self.data)