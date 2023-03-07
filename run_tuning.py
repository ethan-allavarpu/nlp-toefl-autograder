import json
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from modeling.model import BaseModel, ETSModel, HierarchicalModel
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
import random
import argparse

from modeling import trainer
from data_loading.datasets import DefaultDataset
from data_loading.dataloaders import get_data_loaders
from settings import *
import run_utils as utils
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

torch.manual_seed(0)
global_args = None

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def load_data(tokenizer):
    dataset = utils.get_dataset(global_args, tokenizer)
    return dataset

def train_finetune(tune_config, filename='best-params'):
    args = global_args  

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, trust_remote_code=True
    )

    dataset = load_data(tokenizer)
    from modeling import trainer
    train_config = trainer.TrainerConfig(
        max_epochs=tune_config["max_epochs"],
        learning_rate=tune_config["lr"],
        lr_decay=tune_config["lr_decay"],
        num_workers=4,
    )
    train_dl, val_dl, _ = get_data_loaders(
        dataset,
        val_size=0.2,
        test_size=0,
        batch_size=tune_config["batch_size"],
        val_batch_size=1,
        test_batch_size=1,
        num_workers=0,
    )
    model = BaseModel(
        seq_length=dataset.tokenizer.model_max_length,
        num_outputs=len(dataset.targets.columns),
        pretrain_model_name=args.tokenizer_name,
    )
    if args.reading_params_path is not None:
        model.load_state_dict(torch.load(args.reading_params_path), strict=False)

    trainer = trainer.Trainer(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        config=train_config
    )

    trainer.tokens = 0 # counter used for learning rate decay
    model_min_loss = float('inf')
    for epoch in range(tune_config["max_epochs"]):
        train_loss = trainer.train('train', epoch)
        if trainer.val_dataloader:
            val_loss = trainer.train('val', epoch)
        else:
            val_loss = None
        
        if (val_loss < model_min_loss):
            model_min_loss = val_loss
            torch.save(model.state_dict(), "/home/ubuntu/nlp-toefl-autograder/tuning/"+filename)
            with open("/home/ubuntu/nlp-toefl-autograder/tuning/"+filename+".txt", 'w') as convert_file:
                convert_file.write(json.dumps(tune_config))

        trainer.losses.append((train_loss, val_loss))
        tune.report(loss=(val_loss))
        
    print("Finished Training")


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2, filename=None):
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1" 

    # Parameters to tune
    tune_config = {
        "lr": tune.loguniform(2e-5, 1e-1),
        "lr_decay": tune.choice([True, False]),
        "max_epochs": tune.choice([5, 10, 15, 20]),
        "batch_size": tune.choice([4, 8, 16])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"]
    )

    result = tune.run(
        partial(train_finetune, filename=filename),
        resources_per_trial={"cpu": 7, "gpu": gpus_per_trial},
        config=tune_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))



if __name__ == "__main__":
    import sys
    sys.stdout.fileno = lambda: False

    # Make args for tuning and set global equal to those args.
    baseline_args = Namespace(
        ICNALE_output="overall",
        dataset="ICNALE-EDITED",
        function="finetune",
        learning_rate=2e-05,
        loss_path="losses.txt",
        lr_decay=False,
        max_epochs=20,
        model_type="base",
        outputs_path="predictions.txt",
        reading_params_path=None,
        tokenizer_name="distilbert-base-uncased",
        writing_params_path="icnale-baseline.params",
    )
    global_args = baseline_args # change this
    params_output_name = "baseline-best-model.params" # change this

    # Before running go to trainer.py and uncomment line#12
    main(num_samples=15, max_num_epochs=25, gpus_per_trial=1, filename=params_output_name)
