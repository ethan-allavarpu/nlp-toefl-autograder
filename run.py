import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from modeling.model import BaseModel
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
import random
import argparse

from modeling import trainer
from data_loading.datasets import DefaultDataset
from data_loading.dataloaders import get_data_loaders
from settings import ELL_DATA_DIR


# argp = argparse.ArgumentParser()
# argp.add_argument('function', help="Choose pretrain, finetune, or evaluate")
# args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'mps'


# instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
# instantiate the dataset
dataset = DefaultDataset(file_path=ELL_DATA_DIR, input_col='full_text', target_cols=['cohesion'], index_col='text_id', 
                             tokenizer=tokenizer)
                             
# get the dataloaders. can make test and val sizes 0 if you don't want them
train_dl, val_dl, test_dl = get_data_loaders(dataset, val_size=0.2, test_size=0.2, batch_size=4, val_batch_size=1,
    test_batch_size=1, num_workers=0)
# TensorBoard training log
writer = SummaryWriter(log_dir='expt/')

train_config = trainer.TrainerConfig(max_epochs=650, 
        batch_size=128, 
        # learning_rate=args.pretrain_lr, 
        learning_rate=0.01, 
        num_workers=4, writer=writer)
model = BaseModel(num_outputs=1)
trainer = trainer.Trainer(model, train_dl, None, train_config)
trainer.train()
torch.save(model.state_dict(), args.writing_params_path)
