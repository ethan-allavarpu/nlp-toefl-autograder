"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
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

    def __init__(self, model, train_dataloader, config, val_dataloader=None, test_dataloader=None, one_output=False):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.val_dataloader = val_dataloader
        self.losses = []
        self.optimizer = self.create_optimizer()
        # flag for speech
        self.one_output = one_output

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

    def create_optimizer(self):
        model, config = self.model, self.config

        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": config.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)
        return optimizer

    def train(self, split, step):
        model, config = self.model, self.config
        is_train = split == 'train'
        model.train(is_train)
        loader = self.train_dataloader if is_train else self.val_dataloader

        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        losses = []
        for it, (x, y) in pbar:
            # place data on the correct device
            x = x.to(self.device)
            if type(y) == list:
                y = [yy.to(self.device) for yy in y]
            else:
                y = y.to(torch.float32).to(self.device)

            # forward the model
            if is_train:
                model.train()
            else:
                model.eval()
            with torch.set_grad_enabled(is_train):
                logits, loss = model(x, y, val=(not is_train), one_output=self.one_output)
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())
            if is_train:
                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()

                lr = config.learning_rate
                # decay the learning rate based on our progress
                if config.lr_decay:
                    if type(y) == list:
                        self.tokens += (y[0] >= 0).sum()
                    else:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                    if self.tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate
                # report progress
                pbar.set_description(f"epoch {step+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                
                if config.writer is not None:
                    config.writer.add_scalar('train/loss',  loss.item(), step)
                    config.writer.add_scalar('train/lr', lr, step)
                    
        return np.mean(losses)
