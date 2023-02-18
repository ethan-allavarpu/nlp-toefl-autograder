import pandas as pd
import torch
from typing import Sequence, Any, Dict, Tuple, TypeVar
import numpy as np

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
        self.tokenizer_params = tokenizer_params if tokenizer_params else {'padding': 'max_length', 'max_length': max(self.data[input_col].str.len()),
         'truncation': True} 
    
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




# import pandas as pd
# import torch
# from typing import Sequence, Any, Dict, Tuple, TypeVar
# import numpy as np

# T_co = TypeVar('T_co', covariant=True)

# class DefaultDataset(torch.utils.data.Dataset):
#     def __init__(self, file_path, input_col: str, target_cols: Sequence[str], index_col: str = None,
#                  tokenizer: Any = None, tokenizer_params: Dict = None):
#         self.data = pd.read_csv(file_path)
#         self.data.dropna(subset=[input_col], inplace=True)
#         # set index col so we can use it as a key
#         self.data.set_index(index_col, inplace=True)
#         self.indices = self.data.index.tolist()

#         # self.inputs = pd.DataFrame(self.data[input_col])
#         self.targets = pd.DataFrame(self.data[target_cols])
#         # normalize the targets
#         self.normalize_targets()
#         # here we can provide the tokenizer!
#         self.tokenizer = tokenizer
#         # default tokenizer params
#         self.tokenizer_params = tokenizer_params if tokenizer_params else {'padding': True, 'truncation': True}
#         # tokenize the inputs, make into dict so we retain the index
#         self.inputs = dict(self.tokenizer(self.data[input_col].astype(str).values.tolist(), **self.tokenizer_params))
#         self.inputs = {idx: 
#         {'input_ids' : self.inputs['input_ids'][i], 'attention_mask' : self.inputs['attention_mask'][i]} 
#         for i, idx in enumerate(self.indices)}
    
#     def normalize_targets(self, normalize_score: float = 100.0) -> None:
#         self.targets = (self.targets) / self.targets.max(axis=0) * normalize_score

#     def __getitem__(self, index: Any) -> T_co:
#         idx = self.indices[index]
#         return self.inputs[idx], self.targets.loc[idx].values

#     def __len__(self) -> int:
#         return len(self.data)