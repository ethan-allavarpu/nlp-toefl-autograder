import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn import functional as F
from typing import Any

class BaseModel(torch.nn.Module):
    def __init__(self, seq_length: int, num_outputs: int, pretrain_model_name: str):
        super(BaseModel, self).__init__()
        self.seq_length = seq_length
        self.l1 = AutoModel.from_pretrained(pretrain_model_name, trust_remote_code=True)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(self.seq_length*768, num_outputs)
    
    def forward(self, data: Any, targets: Any = None, one_output: bool = True):
        output_1= self.l1(input_ids=data['input_ids'].squeeze(1), attention_mask = data['attention_mask'].squeeze(1))
        # output_2 = self.l2(output_1['pooler_output'])
        output_2 = self.l2(output_1['last_hidden_state'].reshape(-1, self.seq_length*768))
        output = self.l3(output_2)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = nn.MSELoss()(output.float(), targets.float())
        return output, loss

class HierarchicalModel(torch.nn.Module):
    def __init__(self, seq_length: int, num_outputs: int, pretrain_model_name: str):
        super(HierarchicalModel, self).__init__()
        self.seq_length = seq_length
        self.l1 = AutoModel.from_pretrained(pretrain_model_name, trust_remote_code=True)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(self.seq_length*768, num_outputs)
        self.l4 = torch.nn.Linear(num_outputs, 1)
    
    def forward(self, data: Any, targets: Any = None, one_output: bool=True):
        x1 = self.l1(input_ids=data['input_ids'].squeeze(1), attention_mask = data['attention_mask'].squeeze(1))
        x2 = self.l2(x1['last_hidden_state'].reshape(-1, self.seq_length*768))
        output_1 = self.l3(x2)
        output_2 = self.l4(output_1)
        if one_output:
            output = output_2
        else:
            output = output_1
        loss = None
        if targets is not None:
            loss = nn.MSELoss()(output.float(), targets.float())
        return output, loss

class SpeechModel(torch.nn.Module):
    def __init__(self, num_outputs: int, pretrain_model_name: str):
        super(SpeechModel, self).__init__()
        self.l1 = AutoModel.from_pretrained(pretrain_model_name, trust_remote_code=True)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(49*768, num_outputs)
    
    def forward(self, data: Any, targets: Any = None, one_output: bool = True):
        output_1= self.l1(data)
        # output_2 = self.l2(output_1['pooler_output'])
        output_2 = output_1['last_hidden_state'].reshape(-1, 49*768)
        output = self.l3(output_2)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = nn.MSELoss()(output.float(), targets.float())
        return output, loss