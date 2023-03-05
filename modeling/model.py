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
        multi_out = self.l3(x2)
        single_out = self.l4(multi_out)
        if one_output:
            output = single_out
        else:
            output = multi_out

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
        self.phoneme_lstm = torch.nn.LSTM(49*768, 30, bidirectional=True, batch_first=True)
        self.phoneme_fc = torch.nn.Linear(60, 30)
        self.word_fc = torch.nn.Linear(60, 10*3)
        self.output_fc = torch.nn.Linear(30*2, num_outputs)
    
    def forward(self, data: Any, targets: Any = None, one_output: bool = True):
        if type(targets) != list:
            output_1= self.l1(data)
            output_2 = output_1['last_hidden_state'].reshape(-1, 49*768)
            output = self.l3(output_2)
        else:
            output_1= self.l1(data)
            output_2 = output_1['last_hidden_state'].reshape(-1, 49*768)

            phoneme_output, (hn, cn) = self.phoneme_lstm(output_2.unsqueeze(dim=1))
            phoneme_fc = self.phoneme_fc(phoneme_output.squeeze(dim=1)).view(-1, 30)

       
            word_fc = self.word_fc(phoneme_output).view(-1, 10 * 3)
            
            output = self.output_fc(torch.permute(hn, (1, 0, 2)).reshape(-1, 30*2))

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            if type(targets) != list:
                loss = nn.MSELoss()(output.float(), targets.float())
                word_loss = 0
                phoneme_loss = 0
            else:
                loss = nn.MSELoss()(output.float(), targets[0].float())
                # mask out the loss for the padding
                word_targets = targets[1].float().reshape(-1, 10*3)
                word_loss = nn.MSELoss(reduction='none')(word_fc.float(), word_targets)
                word_loss = word_loss[word_targets!=-1].sum()/word_loss[word_targets!=-1].shape[0]

                phoneme_targets = targets[2].float().reshape(-1, 30*1)
                phoneme_loss = nn.MSELoss(reduction='none')(phoneme_fc.float(), phoneme_targets)
                phoneme_loss = phoneme_loss[phoneme_targets!=-1].sum()/phoneme_loss[phoneme_targets!=-1].shape[0]
                
        return output, (loss + word_loss + phoneme_loss)


class GranularSpeechModel(torch.nn.Module):
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
            loss = nn.MSELoss()(output.float(), targets[0].float())
        return output, loss