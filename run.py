import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from modeling.model import BaseModel, HierarchicalModel
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
import random
import argparse

from modeling import trainer
from data_loading.datasets import DefaultDataset
from data_loading.dataloaders import get_data_loaders
from settings import *

torch.manual_seed(0)
argp = argparse.ArgumentParser()
argp.add_argument('function', help="Choose pretrain, finetune, or evaluate") #TODO: add behavior for pretrain and eval
argp.add_argument("--model_type", type=str, default="base", required=False)
argp.add_argument('--writing_params_path', type=str, help='Path to the writing params file', required=False)
argp.add_argument('--reading_params_path', type=str, help='Path to the reading params file', required=False)
argp.add_argument('--loss_path', type=str, help='Path to the output losses', default="losses.txt", required=False)
argp.add_argument('--outputs_path', type=str, help='Path to the output predictions', default="predictions.txt", required=False)
argp.add_argument('--tokenizer_name', type=str, help='Name of the tokenizer to use', default="distilbert-base-uncased", required=False)
argp.add_argument('--dataset', type=str, help='Name of the dataset to use', default="ICNALE-EDITED", required=False)
argp.add_argument("--ICNALE_output", type=str, help="Use 'categories' or 'overall' score", default="overall", required=False)
argp.add_argument('--max_epochs', type=int, help='Number of epochs to train for', default=20, required=False)
argp.add_argument('--learning_rate', type=float, help='Learning rate', default=2e-5, required=False)
args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'


# instantiate the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("ccdv/lsg-xlm-roberta-base-4096", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
# instantiate the dataset
if args.model_type == "base":
    if args.dataset == "ELL":
        dataset = DefaultDataset(file_path=ELL_DATA_DIR, input_col='full_text', target_cols=['cohesion', 'syntax',  'vocabulary',  'phraseology',  'grammar',  'conventions'], index_col='text_id', 
                                tokenizer=tokenizer)
    elif args.dataset == "ICNALE-EDITED":
        if args.ICNALE_output == "overall":
            dataset = DefaultDataset(file_path=ICNALE_EDITED_DATA_DIR, input_col='essay', target_cols=['Total 1 (%)'], index_col=None, 
                                    tokenizer=tokenizer)
        else:
            dataset = DefaultDataset(file_path=ICNALE_EDITED_DATA_DIR, input_col='essay', target_cols=['Content (/12)', 'Organization (/12)',
        'Vocabulary (/12)', 'Language Use (/12)', 'Mechanics (/12)'], index_col=None, 
                                    tokenizer=tokenizer)
    elif args.dataset == "ICNALE-WRITTEN": #TODO: lots of missing fields in this dataset, what imputations should we use?
        dataset = DefaultDataset(file_path=ICNALE_WRITTEN_DATA_DIR, input_col='essay', target_cols=['Score'], index_col="sortkey", 
                                tokenizer=tokenizer)
    elif args.dataset == "FCE":
        dataset = DefaultDataset(file_path=FCE_DATA_DIR, input_col='essay', target_cols=['overall_score'], 
                                tokenizer=tokenizer)
    else:
        raise ValueError("Invalid dataset name")
elif args.model_type == "hierarchical":
    if args.dataset == "ELL-ICNALE":
        dataset1 = DefaultDataset(file_path=ELL_DATA_DIR, input_col='full_text', target_cols=['cohesion', 'syntax',  'vocabulary',  'phraseology',  'grammar',  'conventions'], index_col='text_id', 
                                tokenizer=tokenizer)
        dataset2 = DefaultDataset(file_path=ICNALE_EDITED_DATA_DIR, input_col='essay', target_cols=['Total 1 (%)'], index_col=None, 
                                    tokenizer=tokenizer)
    elif args.dataset == "FCE":
        dataset = DefaultDataset(file_path=FCE_DATA_DIR, input_col='essay', target_cols=['overall_score'], 
                                tokenizer=tokenizer)
    else:
        raise ValueError("Invalid dataset name")

if args.function == 'pretrain':
    pass

elif args.function == 'finetune':
    # TensorBoard training log
    writer = SummaryWriter(log_dir='expt/')

    train_config = trainer.TrainerConfig(max_epochs=args.max_epochs, 
            learning_rate=args.learning_rate, 
            num_workers=4, writer=writer, ckpt_path='expt/params.pt')
    # get the dataloaders. can make test and val sizes 0 if you don't want them
    if args.model_type == "base":
        train_dl, val_dl, test_dl = get_data_loaders(dataset, val_size=0, test_size=0.2, batch_size=16, val_batch_size=1,
        test_batch_size=1, num_workers=0)
    
        model = BaseModel(seq_length=dataset.tokenizer.model_max_length, num_outputs=len(dataset.targets.columns), pretrain_model_name=args.tokenizer_name)
        trainer = trainer.Trainer(model=model,  train_dataloader=train_dl, test_dataloader=test_dl, config=train_config, val_dataloader=None)
        with open(args.loss_path, 'w') as f:
            for loss in trainer.losses:
                f.write(f"{loss[0]},{loss[1]}\n")
    
    elif args.model_type == "hierarchical":
        train_dl1, val_dl1, test_dl1 = get_data_loaders(dataset1, val_size=0, test_size=0.2, batch_size=16, val_batch_size=1,
        test_batch_size=1, num_workers=0)
        train_dl2, val_dl2, test_dl2 = get_data_loaders(dataset2, val_size=0, test_size=0.2, batch_size=16, val_batch_size=1,
        test_batch_size=1, num_workers=0)

        model = HierarchicalModel(seq_length=dataset1.tokenizer.model_max_length, num_outputs=len(dataset1.targets.columns), pretrain_model_name=args.tokenizer_name)
        
        trainer = trainer.HierarchicalTrainer(model, train_dl1, train_dl2, test_dl1, test_dl2, train_config)

    trainer.train()
    torch.save(model.state_dict(), args.writing_params_path)

elif args.function == 'evaluate':
    if args.dataset == "FCE":
        test_dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,num_workers=0)
    else:
        train_dl, val_dl, test_dl = get_data_loaders(dataset, val_size=0.2, test_size=0.0, batch_size=16, val_batch_size=1,
            test_batch_size=1, num_workers=0)
    model = BaseModel(seq_length=dataset.tokenizer.model_max_length, num_outputs=len(dataset.targets.columns), pretrain_model_name=args.tokenizer_name)
    if args.dataset == "FCE" and "ell" in args.reading_params_path:
        model = BaseModel(seq_length=dataset.tokenizer.model_max_length, num_outputs=6, pretrain_model_name=args.tokenizer_name)
    if args.dataset == "FCE" and "icnale-baseline-categories" in args.reading_params_path:
        model = BaseModel(seq_length=dataset.tokenizer.model_max_length, num_outputs=5, pretrain_model_name=args.tokenizer_name)
    if args.model_type == "hierarchical":
        model = HierarchicalModel(seq_length=dataset.tokenizer.model_max_length, num_outputs=6, pretrain_model_name=args.tokenizer_name)
    model.load_state_dict(torch.load(args.reading_params_path))
    model = model.to(device)
    model.eval()
    predictions = []

    pbar = tqdm(enumerate(test_dl), total=len(test_dl)) 
    for it, (x, y) in pbar:
        # place data on the correct device
        x = x.to(device)
        predictions.append((model(x)[0].mean().item(), y[0].mean().item()))
        torch.cuda.empty_cache()

    with open(args.outputs_path, 'w') as f:
        for pred in predictions:
            f.write(f"{pred[0]},{pred[1]}\n")
    

else:
    print("Invalid function name. Choose pretrain, finetune, or evaluate")
