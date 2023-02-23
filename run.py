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
from settings import ELL_DATA_DIR, FCE_DATA_DIR, ETS_DATA_DIR, ICNALE_EDITED_DATA_DIR, ICNALE_WRITTEN_DATA_DIR


argp = argparse.ArgumentParser()
argp.add_argument('function', help="Choose pretrain, finetune, or evaluate") #TODO: add behavior for pretrain and eval
argp.add_argument('--writing_params_path', type=str, help='Path to the writing params file', default="writing_params.json", required=False)
argp.add_argument('--tokenizer_name', type=str, help='Name of the tokenizer to use', default="distilbert-base-uncased", required=False)
argp.add_argument('--dataset', type=str, help='Name of the dataset to use', default="ICNALE-EDITED", required=False)
argp.add_argument('--max_epochs', type=int, help='Number of epochs to train for', default=500, required=False)
argp.add_argument('--learning_rate', type=float, help='Learning rate', default=0.01, required=False)
args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'


# instantiate the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("ccdv/lsg-xlm-roberta-base-4096", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
# instantiate the dataset
if args.dataset == "ELL":
    dataset = DefaultDataset(file_path=ELL_DATA_DIR, input_col='full_text', target_cols=['cohesion', 'syntax',  'vocabulary',  'phraseology',  'grammar',  'conventions'], index_col='text_id', 
                             tokenizer=tokenizer)
elif args.dataset == "ICNALE-EDITED":
    dataset = DefaultDataset(file_path=ICNALE_EDITED_DATA_DIR, input_col='essay', target_cols=['Total 1 (%)'], index_col=None, 
                             tokenizer=tokenizer)
elif args.dataset == "ICNALE-WRITTEN": #TODO: lots of missing fields in this dataset, what imputations should we use?
    dataset = DefaultDataset(file_path=ICNALE_WRITTEN_DATA_DIR, input_col='essay', target_cols=['Score'], index_col=None, 
                             tokenizer=tokenizer)
else:
    raise ValueError("Invalid dataset name")
                             

if args.function == 'pretrain':
    pass

elif args.function == 'finetune':
    max_epochs = 20
    # get the dataloaders. can make test and val sizes 0 if you don't want them
    train_dl, val_dl, test_dl = get_data_loaders(dataset, val_size=0, test_size=0.2, batch_size=16, val_batch_size=1,
        test_batch_size=1, num_workers=0)
    # TensorBoard training log
    writer = SummaryWriter(log_dir='expt/')

    train_config = trainer.TrainerConfig(max_epochs=max_epochs, 
            learning_rate=args.learning_rate, 
            num_workers=4, writer=writer, ckpt_path='expt/params.pt')

    model = BaseModel(num_outputs=len(dataset.targets.columns), pretrain_model_name=args.tokenizer_name)
    trainer = trainer.Trainer(model, train_dl, test_dl, train_config)
    trainer.train()
    torch.save(model.state_dict(), args.writing_params_path)

if args.function == 'evaluate':
    pass
