import pandas as pd
import torch
from typing import Sequence, Any, Dict, Tuple, TypeVar
import numpy as np
from datasets import load_dataset, Dataset
from torch.nn.functional import pad

T_co = TypeVar('T_co', covariant=True)

def pad_up_to(t, max_in_dims, constant_values):
    s = t.shape
    paddings = (m-s[i] for (i,m) in enumerate(max_in_dims))
    return F.pad(t, paddings, 'constant', constant_values=constant_values)

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
    def __init__(self, path_name: str, input_col: str, target_cols_sentence: Sequence[str], 
    target_cols_words: Sequence[str] = [], target_cols_phones: Sequence[str] = [],
                 tokenizer: Any = None, tokenizer_params: Dict = None):
        self.data = load_dataset(path_name, split='train')
        self.tokenizer = tokenizer

        tokenizer_params = tokenizer_params if tokenizer_params else {'sampling_rate':tokenizer.sampling_rate, 'max_length': tokenizer.sampling_rate, 
        'truncation': True}
        self.inputs = tokenizer(
        [x[input_col]['array'] for x in self.data], ** tokenizer_params)['input_values']

        self.targets_sentence = pd.DataFrame([[x[t] for t in target_cols_sentence] for x in self.data ], columns=target_cols_sentence)
        if (len(target_cols_words) > 0) | (len(target_cols_phones) > 0):
            targets_words, targets_phones = self.compute_subtargets(target_cols_words, target_cols_phones)
        
        if len(target_cols_words) > 0:
            self.targets_words = pd.DataFrame(targets_words, columns=target_cols_words)
        else:
            self.targets_words = None
        if len(target_cols_phones) > 0:
            self.targets_phones = pd.DataFrame(targets_phones, columns=target_cols_phones)
        else:
            self.targets_phones = None
        
        # normalize the targets
        self.normalize_targets()
    
    def compute_subtargets(self, target_cols_words: Sequence[str], target_cols_phones: Sequence[str]) -> Tuple[Sequence[str], Sequence[str]]:
        compute_phone_accuracy = len(target_cols_phones) > 0
        compute_word_accuracy = len(target_cols_words) > 0
        targets_words = []
        targets_phones = []
        for person_words in self.data['words']:
            person_word_list = []
            person_phones_list = []
            for w in target_cols_words:
                word_scores = []
                for word_dict in person_words:
                    word_scores.append(word_dict[w])
                person_word_list.append(word_scores)
            for p in target_cols_phones:
                word_phone_scores = []
                for word_dict in person_words:
                    # word_phone_scores.append(np.array(word_dict[p]))
                    word_phone_scores.append(word_dict[p])
                # person_phones_list.append(word_phone_scores)
                person_phones_list.append([score for sublist in word_phone_scores for score in sublist])
            if compute_word_accuracy:
                targets_words.append(person_word_list)
            if compute_phone_accuracy:
                targets_phones.append(person_phones_list)
        return targets_words, targets_phones

    def normalize_targets(self, normalize_score: float = 100.0) -> None:
        # normalize the sentence targets
        self.targets_sentence = (self.targets_sentence) / self.targets_sentence.max(axis=0) * normalize_score
        # normalize the word targets
        if self.targets_words is not None:
            for label in range(len(self.targets_words.iloc[0])):
                # loop through each word target and normalize it
                def normalize_score(row, max_score: float):
                    row[label] = [score/max_score*100 for score in row[label]]
                    return row
                max_score = self.targets_words.apply(lambda x: max(x[label]), axis=1).max()
                self.targets_words = self.targets_words.apply(normalize_score, max_score=max_score, axis=1)
        # normalize the phone targets
        if self.targets_phones is not None:
            for label in range(len(self.targets_phones.iloc[0])):
                # loop through each word target and normalize it
                def normalize_score(row, max_score: float):
                    # row[label] = np.array([np.array([score/max_score*100 for score in phone]) for phone in row[label]])
                    row[label] = [score/max_score*100 for score in row[label]]
                    return row
                # max_score = self.targets_phones.apply(lambda x: max(list(map(max, x[label]))), axis=1).max()
                max_score = self.targets_words.apply(lambda x: max(x[label]), axis=1).max()
                self.targets_phones = self.targets_phones.apply(normalize_score, max_score =max_score, axis=1)

    def __getitem__(self, index: Any) -> T_co:
        words_output = None if self.targets_words is None else np.array(self.targets_words.loc[index].values.tolist())
        if words_output is not None:
            words_output = torch.Tensor(words_output)
            words_output = pad(words_output, (0, 10-words_output.shape[1], 0, 0), value=-1)
        phones_output = None if self.targets_phones is None else np.array(self.targets_phones.loc[index].values.tolist())
        if phones_output is not None:
            phones_output = torch.Tensor(phones_output)
            phones_output = pad(phones_output, (0, 30-phones_output.shape[1], 0, 0), value=-1)
        return self.inputs[index], [self.targets_sentence.loc[index].values, words_output, phones_output]

            
    def __len__(self) -> int:
        return len(self.data)