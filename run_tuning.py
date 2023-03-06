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


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def train_finetune(tune_config, args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, trust_remote_code=True
    )
    dataset = utils.get_dataset(args, tokenizer)
    train_config = trainer.TrainerConfig(
        max_epochs=tune_config["max_epochs"],
        learning_rate=tune_config["lr"],
        lr_decay=tune_config["lr_decay"],
        num_workers=4,
        ckpt_path="tuning/params.pt",
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
        config=train_config,
        val_dataloader=None,
    )

    trainer.train()




def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):

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
        grace_period=1,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        # parameter_columns=["lr", "lr_decay", "max_epochs"],
        metric_columns=["loss", "accuracy", "training_iteration"]
    )

    result = tune.run(
        partial(train_finetune, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=tune_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print(
        "Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]
        )
    )



if __name__ == "__main__":
    # sphinx_gallery_start_ignore
    # Fixes AttributeError: '_LoggingTee' object has no attribute 'fileno'.
    # This is only needed to run with sphinx-build.
    import sys

    sys.stdout.fileno = lambda: False
    # sphinx_gallery_end_ignore
    # You can change the number of GPUs per trial here:
    args = Namespace(
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
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
