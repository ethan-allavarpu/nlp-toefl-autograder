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
        self.l3 = torch.nn.Linear(self.seq_length*768, self.seq_length)
        self.l4 = torch.nn.ReLU()
        self.l5 = torch.nn.Linear(self.seq_length, num_outputs)

        
    def forward(self, data: Any, targets: Any = None, eval_output: bool = False):
        output_1= self.l1(input_ids=data['input_ids'].squeeze(1), attention_mask = data['attention_mask'].squeeze(1))
        # output_2 = self.l2(output_1['pooler_output'])
        output_2 = self.l2(output_1['last_hidden_state'].reshape(-1, self.seq_length*768))
        output_3 = self.l3(output_2)
        output_4 = self.l4(output_3)
        output = self.l5(output_4)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = nn.MSELoss()(output.float(), targets.float())
        return output, loss

class ETSModel(torch.nn.Module):
    def __init__(self, seq_length: int, num_outputs: int, pretrain_model_name: str):
        super(ETSModel, self).__init__()
        self.seq_length = seq_length
        self.l1 = AutoModel.from_pretrained(pretrain_model_name, trust_remote_code=True)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(self.seq_length*768, self.seq_length)
        self.relu_layer = torch.nn.ReLU()
        self.final = torch.nn.Linear(self.seq_length, num_outputs)
    
    def forward(self, data: Any, targets: Any = None, eval_output: bool = False):
        output_1= self.l1(input_ids=data['input_ids'].squeeze(1), attention_mask = data['attention_mask'].squeeze(1))
        # output_2 = self.l2(output_1['pooler_output'])
        output_2 = self.l2(output_1['last_hidden_state'].reshape(-1, self.seq_length*768))
        output_3 = self.l3(output_2)
        output_relu = self.relu_layer(output_3)
        output = self.final(output_relu)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = nn.CrossEntropyLoss()(output.float(), targets.float())
        return output, loss


class HierarchicalModel(torch.nn.Module):
    def __init__(self, seq_length: int, num_outputs: int, pretrain_model_name: str):
        super(HierarchicalModel, self).__init__()
        self.seq_length = seq_length
        self.l1 = AutoModel.from_pretrained(pretrain_model_name, trust_remote_code=True)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(self.seq_length*768, num_outputs)
        self.l4 = torch.nn.Linear(num_outputs, 1)
    
    def forward(self, data: Any, targets: Any = None, eval_output: bool = False):
        x1 = self.l1(input_ids=data['input_ids'].squeeze(1), attention_mask = data['attention_mask'].squeeze(1))
        x2 = self.l2(x1['last_hidden_state'].reshape(-1, self.seq_length*768))
        multi_out = self.l3(x2)
        single_out = self.l4(multi_out)
        output = torch.cat([multi_out, single_out], dim=1)
        if eval_output:
            output = single_out
        loss = None
        if targets is not None:
            loss = nn.MSELoss()(
                output[targets > -999].float(), targets[targets > -999].float()
            )
        return output, loss

class SpeechModel(torch.nn.Module):
    def __init__(self, num_outputs: int, pretrain_model_name: str, phoneme_seq_length: int, word_seq_length: int, word_outputs: int):
        super(SpeechModel, self).__init__()
        self.l1 = AutoModel.from_pretrained(pretrain_model_name, trust_remote_code=True)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(312*768, num_outputs)
        # self.l4 = torch.nn.Linear(312*768, word_outputs*word_seq_length)
        # self.l5 = torch.nn.Linear(312*768, phoneme_seq_length)
        self.phoneme_lstm = torch.nn.LSTM(768, phoneme_seq_length, bidirectional=True, batch_first=True)
        # mult by 2 bc bidir
        self.phoneme_fc = torch.nn.Linear(phoneme_seq_length*2*312, phoneme_seq_length)
        self.word_fc = torch.nn.Linear(phoneme_seq_length*2*312, word_seq_length*word_outputs)
        self.output_fc = torch.nn.Linear(phoneme_seq_length*2*312, num_outputs)
        # add nonlinearity
        # self.relu_output = torch.nn.ReLU()
        # self.relu_word = torch.nn.ReLU()
        # self.relu_phoneme = torch.nn.ReLU()

        self.word_outputs = word_outputs
        self.word_seq_length = word_seq_length
        self.phoneme_seq_length = phoneme_seq_length
    
    def forward(self, data: Any, targets: Any = None, one_output: bool = True):
        if (len(targets)<3) & (one_output):
            output_1= self.l1(data)
            output_2 = output_1['last_hidden_state'].reshape(-1, 312*768)
            output = self.l3(output_2)
            word_output, phoneme_output = (0,0)
        else:
            output_1= self.l1(data)
            output_2 = output_1['last_hidden_state'].reshape(-1, 312*768)

            lstm_output, (hn, cn) = self.phoneme_lstm(output_1['last_hidden_state'])
            lstm_output = lstm_output.reshape(-1, self.phoneme_seq_length*2*312)
            phoneme_fc = self.phoneme_fc(lstm_output)

       
            word_fc = self.word_fc(lstm_output).view(-1, self.word_outputs * self.word_seq_length)
            
            output = self.output_fc(lstm_output)
            # output = self.l3(output_2)
            # word_fc = self.l4(output_2* self.word_seq_length)
            # phoneme_fc = self.l5(output_2)
            word_output = word_fc.float()
            phoneme_output = phoneme_fc.float()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            if len(targets)<3:
                loss = nn.MSELoss()(output.float(), targets[0].float())
            else:
                loss = nn.MSELoss()(output.float(), targets[0].float())
                # mask out the loss for the padding
                word_targets = targets[1].float().reshape(-1, 10*3)
                word_loss = nn.MSELoss(reduction='none')(word_fc.float(), word_targets)
                word_loss = word_loss[word_targets!=-1].sum()/word_loss[word_targets!=-1].shape[0]

                phoneme_targets = targets[2].float().reshape(-1, 30*1)
                phoneme_loss = nn.MSELoss(reduction='none')(phoneme_fc.float(), phoneme_targets)
                phoneme_loss = phoneme_loss[phoneme_targets!=-1].sum()/phoneme_loss[phoneme_targets!=-1].shape[0]
                loss = loss + word_loss + phoneme_loss
        return (output, word_output, phoneme_output), loss


class SiameseSpeechModel(torch.nn.Module):
    def __init__(self, num_outputs: int, pretrain_model_name: str, phoneme_seq_length: int, word_seq_length: int, word_outputs: int):
        super(SiameseSpeechModel, self).__init__()
        self.l1 = AutoModel.from_pretrained(pretrain_model_name, trust_remote_code=True)
        self.l2 = torch.nn.Dropout(0.3)
        self.siamese_linear_basic = torch.nn.Linear(312*768*2, num_outputs)
        self.l3 = torch.nn.Linear(312*768, num_outputs)
        # self.l4 = torch.nn.Linear(312*768, word_outputs*word_seq_length)
        # self.l5 = torch.nn.Linear(312*768, phoneme_seq_length)
        self.phoneme_lstm = torch.nn.LSTM(768, phoneme_seq_length, bidirectional=True, batch_first=True)
        # siamese linear layer to combine the two outputs into one
        self.siamese_relu = torch.nn.ReLU()
        self.siamese_linear_phoneme =  torch.nn.Linear(phoneme_seq_length*2*312*2, phoneme_seq_length)
        self.siamese_linear_word = torch.nn.Linear(phoneme_seq_length*2*312*2, word_seq_length*word_outputs)
        self.siamese_linear_output = torch.nn.Linear(phoneme_seq_length*2*312*2, num_outputs)

        # mult by 2 bc bidir
        self.phoneme_fc = torch.nn.Linear(phoneme_seq_length*2*312, phoneme_seq_length)
        self.word_fc = torch.nn.Linear(phoneme_seq_length*2*312, word_seq_length*word_outputs)
        self.output_fc = torch.nn.Linear(phoneme_seq_length*2*312, num_outputs)
        self.word_outputs = word_outputs
        self.word_seq_length = word_seq_length
        self.phoneme_seq_length = phoneme_seq_length
    
    def forward(self, data: Any, targets: Any = None, one_output: bool = True):
        def forward_pass(data, targets, one_output):
            if (len(targets)<3) & (one_output):
                output_1= self.l1(data)
                output_2 = output_1['last_hidden_state'].reshape(-1, 312*768)
                # output = self.l3(output_2)
                # word_output, phoneme_output = (0,0)
                return output_2
            else:
                output_1= self.l1(data)
                output_2 = output_1['last_hidden_state'].reshape(-1, 312*768)

                lstm_output, (hn, cn) = self.phoneme_lstm(output_1['last_hidden_state'])
                # phoneme_fc = self.phoneme_fc(phoneme_output)

        
                # word_fc = self.word_fc(phoneme_output).view(-1, self.word_outputs * self.word_seq_length)
                
                # output = self.output_fc(phoneme_output)
                # word_output = word_fc.float()
                # phoneme_output = phoneme_fc.float()
            return lstm_output.reshape(-1, self.phoneme_seq_length*2*312)

        initial_output = forward_pass(data, targets, one_output)
        # targets[-1] is the siamese data
        correct_output = forward_pass(targets[-1], targets, one_output)

        if (len(targets)<3) & (one_output):
            # TODO: pass through relu?
            siamese_input = torch.cat((initial_output, correct_output), dim=1)
            siamese_output = self.siamese_linear_basic(siamese_input)
            output = self.l3(initial_output)
            word_output, phoneme_output = (0,0)
        else:
            # TODO: pass through relu?
            siamese_input = torch.cat((initial_output, correct_output), dim=1)
            siamese_phoneme_output = self.siamese_linear_phoneme(siamese_input)
            siamese_word_output = self.siamese_linear_word(siamese_input)
            siamese_output = self.siamese_linear_output(siamese_input)

            phoneme_output = self.phoneme_fc(initial_output).float()
            word_output = self.word_fc(initial_output).view(-1, self.word_outputs * self.word_seq_length).float()
            output = self.output_fc(initial_output)
        


        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            if len(targets)<3:
                loss = nn.MSELoss()(output.float(), targets[0].float())
                loss += nn.MSELoss()(siamese_output.float(), targets[0].float())
            else:
                loss = nn.MSELoss()(output.float(), targets[0].float())
                loss += nn.MSELoss()(siamese_output.float(), targets[0].float())
                # mask out the loss for the padding
                word_targets = targets[1].float().reshape(-1, 10*3)
                word_loss = nn.MSELoss(reduction='none')(word_output, word_targets)
                siamese_word_loss = nn.MSELoss(reduction='none')(siamese_word_output, word_targets)
                word_loss = word_loss[word_targets!=-1].sum()/word_loss[word_targets!=-1].shape[0]
                word_loss += siamese_word_loss[word_targets!=-1].sum()/siamese_word_loss[word_targets!=-1].shape[0]

                phoneme_targets = targets[2].float().reshape(-1, 30*1)
                phoneme_loss = nn.MSELoss(reduction='none')(phoneme_output, phoneme_targets)
                siamese_phoneme_loss = nn.MSELoss(reduction='none')(siamese_phoneme_output, phoneme_targets)
                phoneme_loss = phoneme_loss[phoneme_targets!=-1].sum()/phoneme_loss[phoneme_targets!=-1].shape[0]
                phoneme_loss += siamese_phoneme_loss[phoneme_targets!=-1].sum()/siamese_phoneme_loss[phoneme_targets!=-1].shape[0]

                loss = loss + word_loss + phoneme_loss
            
        return (output, word_output, phoneme_output), loss