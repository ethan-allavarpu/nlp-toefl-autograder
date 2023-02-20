import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn import functional as F

class BaseModel(torch.nn.Module):
    def __init__(self, num_outputs: int):
        super(BaseModel, self).__init__()
        self.l1 = AutoModel.from_pretrained('allenai/longformer-base-4096')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, num_outputs)
    
    def forward(self, data, targets = None):
        output_1= self.l1(input_ids=data['input_ids'].squeeze(1), attention_mask = data['attention_mask'].squeeze(1))
        output_2 = self.l2(output_1['pooler_output'])
        output = self.l3(output_2)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = nn.MSELoss()(output.float(), targets.float())
        return output, loss