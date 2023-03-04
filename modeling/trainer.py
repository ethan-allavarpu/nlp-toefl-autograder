"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 1e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    writer = None
    
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataloader, config, val_dataloader=None, test_dataloader=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.val_dataloader = val_dataloader
        self.losses = []

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        if self.config.ckpt_path is not None:
            ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            logger.info("saving %s", self.config.ckpt_path)
            torch.save(ckpt_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config

        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": config.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas)
        step = 0
        def run_epoch(split):
            nonlocal step
            is_train = split == 'train'
            model.train(is_train)
            loader = self.train_dataloader if is_train else self.val_dataloader

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            
            for it, (x, y) in pbar:
                # place data on the correct device
                x = x.to(self.device)
                if type(y) == list:
                    y = [yy.to(self.device) for yy in y]
                else:
                    y = y.to(torch.float32).to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    lr = config.learning_rate
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                    
                    if config.writer is not None:
                        config.writer.add_scalar('train/loss',  loss.item(), step)
                        config.writer.add_scalar('train/lr', lr, step)
                    
                step += 1
            return np.mean(losses)
            if not is_train:
                print("val loss: %f", np.mean(losses))

        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            train_loss = run_epoch('train')
            if self.val_dataloader:
                val_loss = run_epoch('val')
            else:
                val_loss = None
            self.losses.append((train_loss, val_loss))

            self.save_checkpoint()

class HierarchicalTrainer:

    def __init__(self, model, train_dataloader1, train_dataloader2, test_dataloader1, test_dataloader2, config):
        self.model = model
        self.train_dataloader1 = train_dataloader1
        self.test_dataloader1 = test_dataloader1
        self.train_dataloader2 = train_dataloader2
        self.test_dataloader2 = test_dataloader2
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        if self.config.ckpt_path is not None:
            ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            logger.info("saving %s", self.config.ckpt_path)
            torch.save(ckpt_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config

        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": config.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas)
        step = 0
        def run_epoch(split):
            nonlocal step
            is_train = split == 'train'
            model.train(is_train)
            loader1 = self.train_dataloader1 if is_train else self.test_dataloader1
            loader2 = self.train_dataloader2 if is_train else self.test_dataloader2

            losses = []
            pbar1 = tqdm(enumerate(zip(loader1, loader2)), total=len([*enumerate(zip(loader1, loader2))])) if is_train else enumerate(zip(loader1, loader2))
            
            for it, ((x1, y1), (x2, y2)) in pbar1:
                # place data on the correct device
                x1 = x1.to(self.device)
                if type(y1) == list:
                    y1 = [yy.to(self.device) for yy in y1]
                else:
                    y1 = y1.to(torch.float32).to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x1, y1, one_output=False)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    lr = config.learning_rate
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    # report progress
                    pbar1.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                    
                    if config.writer is not None:
                        config.writer.add_scalar('train/loss',  loss.item(), step)
                        config.writer.add_scalar('train/lr', lr, step)
                    
                step += 1

                # place data on the correct device
                x2 = x2.to(self.device)
                if type(y2) == list:
                    y2 = [yy.to(self.device) for yy in y2]
                else:
                    y2 = y2.to(torch.float32).to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x2, y2, one_output=True)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    lr = config.learning_rate
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    # report progress
                    pbar1.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                    
                    if config.writer is not None:
                        config.writer.add_scalar('train/loss',  loss.item(), step)
                        config.writer.add_scalar('train/lr', lr, step)
                    
                step += 1
            if not is_train:
                logger.info("test loss: %f", np.mean(losses))

        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.test_dataloader1:
                run_epoch('test')

            self.save_checkpoint()
