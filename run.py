import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from modeling.model import BaseModel, ETSModel, HierarchicalModel
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
import random

from modeling import trainer
from data_loading.dataloaders import get_data_loaders
import run_utils as utils

torch.manual_seed(0)
args = utils.get_argparser().parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
args.lr_decay = args.lr_decay == "True"

# Instantiate the tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
dataset = utils.get_dataset(args, tokenizer)


if args.function == 'pretrain':
    writer = SummaryWriter(log_dir='expt/')

    train_config = trainer.TrainerConfig(max_epochs=args.max_epochs, 
            learning_rate=args.learning_rate, lr_decay=args.lr_decay,
            num_workers=4, writer=writer, ckpt_path='expt/params.pt')

    if args.model_type == "base":
        train_dl, val_dl, test_dl = get_data_loaders(dataset, val_size=0.2, test_size=0, batch_size=16, val_batch_size=1,
        test_batch_size=1, num_workers=0)

        model = BaseModel(seq_length=dataset.tokenizer.model_max_length, num_outputs=len(dataset.targets.columns), pretrain_model_name=args.tokenizer_name)

        if args.reading_params_path is not None:
            model.load_state_dict(torch.load(args.reading_params_path), strict=False)

        trainer = trainer.Trainer(model=model, train_dataloader=train_dl, test_dataloader=test_dl, config=train_config, val_dataloader=None)
    elif args.model_type == "ets":
        train_dl, val_dl, test_dl = get_data_loaders(dataset, val_size=0.2, test_size=0, batch_size=16, val_batch_size=1,
        test_batch_size=1, num_workers=0)
    
        model = ETSModel(seq_length=dataset.tokenizer.model_max_length, num_outputs=len(dataset.targets.columns), pretrain_model_name=args.tokenizer_name)

        if args.reading_params_path is not None:
            model.load_state_dict(torch.load(args.reading_params_path), strict=False)
        
        trainer = trainer.Trainer(model=model, train_dataloader=train_dl, test_dataloader=test_dl, config=train_config, val_dataloader=None)

    trainer.tokens = 0 # counter used for learning rate decay
    for epoch in range(args.max_epochs):
        train_loss = trainer.train('train', epoch)
        if trainer.val_dataloader:
            val_loss = trainer.train('val', epoch)
        else:
            val_loss = None
        trainer.losses.append((train_loss, val_loss))
        trainer.save_checkpoint()
    
    torch.save(model.state_dict(), args.writing_params_path)


elif args.function == 'finetune':
    # TensorBoard training log
    writer = SummaryWriter(log_dir='expt/')

    train_config = trainer.TrainerConfig(max_epochs=args.max_epochs, 
            learning_rate=args.learning_rate, lr_decay=args.lr_decay,
            num_workers=4, writer=writer, ckpt_path='expt/params.pt')
    # get the dataloaders. can make test and val sizes 0 if you don't want them
    if args.model_type == "base":
        train_dl, val_dl, test_dl = get_data_loaders(dataset, val_size=0.2, test_size=0, batch_size=16, val_batch_size=1,
        test_batch_size=1, num_workers=0)
    
        model = BaseModel(seq_length=dataset.tokenizer.model_max_length, num_outputs=len(dataset.targets.columns), pretrain_model_name=args.tokenizer_name)
        if args.reading_params_path is not None:
            model.load_state_dict(torch.load(args.reading_params_path), strict=False)
        trainer = trainer.Trainer(model=model,  train_dataloader=train_dl, test_dataloader=test_dl, config=train_config, val_dataloader=None)
    
    elif args.model_type == "hierarchical":
        if args.dataset == "ELL-ICNALE":
            train_dl, val_dl, test_dl = get_data_loaders(
                dataset, val_size=0, test_size=0.2, batch_size=32, val_batch_size=1, test_batch_size=1, num_workers=0
            )

            model = HierarchicalModel(seq_length=dataset.tokenizer.model_max_length, num_outputs=len(dataset.targets.columns) - 1, pretrain_model_name=args.tokenizer_name)
            
        elif args.dataset == "ICNALE-EDITED":
            train_dl, val_dl, test_dl = get_data_loaders(dataset, val_size=0.2, test_size=0, batch_size=16, val_batch_size=1,
            test_batch_size=1, num_workers=0)
        
            model = HierarchicalModel(seq_length=dataset.tokenizer.model_max_length, num_outputs=6, pretrain_model_name=args.tokenizer_name)
            if args.reading_params_path is not None:
                model.load_state_dict(torch.load(args.reading_params_path), strict=False)

        trainer = trainer.Trainer(model=model, train_dataloader=train_dl, test_dataloader=test_dl, config=train_config, val_dataloader=None)

    trainer.tokens = 0 # counter used for learning rate decay
    if args.model_type == "base":
        for epoch in range(args.max_epochs):
            train_loss = trainer.train('train', epoch)
            if trainer.val_dataloader:
                val_loss = trainer.train('val', epoch)
            else:
                val_loss = None
            trainer.losses.append((train_loss, val_loss))
            trainer.save_checkpoint()
    else: # hierarchical
        for epoch in range(args.max_epochs):
            train_loss = trainer.train('train', epoch)
            if trainer.test_dataloader1:
                val_loss = trainer.train('test', epoch)
            trainer.save_checkpoint()
    
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

    utils.write_predictions(args.outputs_path, predictions)
    
else:
    print("Invalid function name. Choose pretrain, finetune, or evaluate")
