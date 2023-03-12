import json
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from modeling.model import BaseModel, ETSModel, HierarchicalModel, SpeechModel, SiameseSpeechModel
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoFeatureExtractor
import random
import argparse

from modeling import trainer
from data_loading.datasets import DefaultDataset, SpeechDataset
from data_loading.dataloaders import get_data_loaders, split_on_indices
from settings import *
import run_utils as utils
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

torch.manual_seed(0)
global_args = None;

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def load_data(tokenizer):
    dataset = utils.get_dataset(global_args, tokenizer)
    return dataset

def train_written(tune_config, filename, model_name):
    from modeling import trainer
    args = global_args  

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, trust_remote_code=True
    )

    dataset = load_data(tokenizer)
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
    if model_name == 'baseline':
        model = BaseModel(
            seq_length=dataset.tokenizer.model_max_length,
            num_outputs=len(dataset.targets.columns),
            pretrain_model_name=args.tokenizer_name,
        )
    elif model_name == 'ets':
          model = ETSModel(seq_length=dataset.tokenizer.model_max_length, num_outputs=len(dataset.targets.columns), pretrain_model_name=args.tokenizer_name)
    elif model_name == 'hierarchical':
         model = model = HierarchicalModel(seq_length=dataset.tokenizer.model_max_length, num_outputs=len(dataset.targets.columns) - 1, pretrain_model_name=args.tokenizer_name)
    
    if args.reading_params_path is not None:
        model.load_state_dict(torch.load(args.reading_params_path), strict=False)

    trainer = trainer.Trainer(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        config=train_config
    )

    output_folder = "/home/ubuntu/nlp-toefl-autograder/tuning/{}/trial_{}/".format(model_name, tune.get_trial_id().split('_')[1])
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    trainer.tokens = 0 # counter used for learning rate decay
    model_min_loss=float('inf')
    for epoch in range(tune_config["max_epochs"]):
        train_loss = trainer.train('train', epoch)
        if trainer.val_dataloader:
            val_loss = trainer.train('val', epoch)
        else:
            val_loss = None

        if (val_loss < model_min_loss):

            model_min_loss = val_loss
            tune_config['stopping_epoch'] = epoch
            tune_config['loss'] = val_loss

            torch.save(model.state_dict(), output_folder+"best-model.params")
            with open(output_folder+"best-model.txt", 'w') as convert_file:
                convert_file.write(json.dumps(tune_config))

        trainer.losses.append((train_loss, val_loss))
        with open(output_folder+"all-losses.txt", 'w') as convert_file:
                convert_file.write(json.dumps(trainer.losses))
        tune.report(loss=(val_loss))
    torch.save(model.state_dict(), output_folder+"final-model.params")
    print("Finished Training")

def train_speech(tune_config, model_name, filename='best-params'):
    args = global_args  

    tokenizer = AutoFeatureExtractor.from_pretrained(args.tokenizer_name)

    dataset = SpeechDataset(path_name=SPEECHOCEAN_DATA_DIR, input_col = 'audio', target_cols_sentence=['accuracy', 'fluency', 'prosodic', 'total'],
    target_cols_words = ["accuracy", "stress", "total"], target_cols_phones = ['phones-accuracy'], tokenizer=tokenizer)

    from modeling import trainer
    train_config = trainer.TrainerConfig(
        max_epochs=tune_config["max_epochs"],
        learning_rate=tune_config["lr"],
        lr_decay=tune_config["lr_decay"],
        num_workers=4,
    )
    train_dl, val_dl, _ = split_on_indices(
        dataset,
        index_col="speaker_id",
        val_size=0.1,
        test_size=0.1,
        batch_size=tune_config["batch_size"],
        val_batch_size=16,
        test_batch_size=1,
        num_workers=0,
    )
    use_mod = SpeechModel if model_name == "speech" else SiameseSpeechModel
    model = use_mod(num_outputs=len(dataset.targets_sentence.columns), pretrain_model_name=args.tokenizer_name,
    phoneme_seq_length=dataset.phoneme_seq_length, word_seq_length=dataset.word_seq_length, word_outputs = 0 if dataset.targets_words is None else len(dataset.targets_words.columns))

    if args.reading_params_path is not None:
        model.load_state_dict(torch.load(args.reading_params_path), strict=False)

    trainer = trainer.Trainer(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        config=train_config
    )

    output_folder = "/home/ubuntu/nlp-toefl-autograder/tuning/{}/trial_{}/".format(model_name, tune.get_trial_id().split('_')[1])
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    trainer.tokens = 0 # counter used for learning rate decay
    model_min_loss=float('inf')
    for epoch in range(tune_config["max_epochs"]):
        train_loss = trainer.train('train', epoch)
        if trainer.val_dataloader:
            val_loss = trainer.train('val', epoch)
        else:
            val_loss = None

        if (val_loss < model_min_loss):

            model_min_loss = val_loss
            tune_config['stopping_epoch'] = epoch
            tune_config['loss'] = val_loss

            torch.save(model.state_dict(), output_folder+"best-model.params")
            with open(output_folder+"best-model.txt", 'w') as convert_file:
                convert_file.write(json.dumps(tune_config))

        trainer.losses.append((train_loss, val_loss))
        with open(output_folder+"all-losses.txt", 'w') as convert_file:
                convert_file.write(json.dumps(trainer.losses))
        tune.report(loss=(val_loss))
    torch.save(model.state_dict(), output_folder+"final-model.params")
    print("Finished Training")

def main(model_name, num_samples=15, max_num_epochs=20, gpus_per_trial=1, filename=None):
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1" 

    # Parameters to tune
    if model_name in ["speech", "siamese-speech"]:
         tune_config = {
            "lr": tune.loguniform(2e-5, 2e-5),
            "lr_decay": tune.choice([False]),
            "max_epochs": tune.choice([35]),
            "batch_size": tune.choice([16]),
            "model_name" : model_name
        }
    else:
         tune_config = {
            # "lr": tune.loguniform(2e-5, 1e-1),
            # "lr_decay": tune.choice([True, False]),
            # "max_epochs": tune.choice([25]),
            # "batch_size": tune.choice([4, 8, 16]),
            # "model_name" : model_name
            "lr": tune.loguniform(1e-6, 1e-4),
            "lr_decay": tune.choice([True, False]),
            "max_epochs": tune.choice([30]),
            "batch_size": tune.choice([16, 32]),
            "model_name" : model_name
        }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=max_num_epochs,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"]
    )
    
    if model_name in ["speech", "siamese-speech"]:
        result = tune.run(
            partial(train_speech, filename=filename, model_name=model_name),
            resources_per_trial={"cpu": 7, "gpu": gpus_per_trial},
            config=tune_config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter
        )
    else:
        result = tune.run(
            partial(train_written, filename=filename, model_name = model_name),
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
    argp = argparse.ArgumentParser()
    argp.add_argument('--model', type=str, help='Choose speech/ets/hierarchical/baseline', required=True)
    clargs = argp.parse_args()

    import sys
    sys.stdout.fileno = lambda: False

    # Make args for tuning and set global equal to those args.
    speech_args = Namespace(
        tokenizer_name = "facebook/wav2vec2-base",
        dataset = "SPEECHOCEAN",
        model_type = "speech",
        reading_params_path=None
    )
    ets_args = Namespace(
        ICNALE_output="overall",
        dataset="ETS",
        function="pretrain",
        model_type="ets",
        reading_params_path=None,
        tokenizer_name="distilbert-base-uncased",
        writing_params_path="double-pretrain-ets1.params"
        )
    hierarchical_args = Namespace(
        ICNALE_output="overall",
        dataset="ELL-ICNALE",
        function="finetune",
        model_type="hierarchical",
        reading_params_path=None,
        tokenizer_name="distilbert-base-uncased",
        writing_params_path="hierarchical-model-1.params"
    )
    baseline_args = Namespace(
        ICNALE_output="overall",
        dataset="ICNALE-EDITED",
        function="finetune",
        model_type="base",
        reading_params_path=None,
        tokenizer_name="distilbert-base-uncased",
        writing_params_path="icnale-baseline.params"
    )
    

    if clargs.model == 'speech':
         global_args = speech_args
         params_output_name = "speech-best-model.params"
         trials, epochs_per_trial  = 1, 70
    elif clargs.model == 'siamese-speech':
         global_args = speech_args
         params_output_name = "siamese-speech-best-model.params"
         trials, epochs_per_trial  = 15, 70
    elif clargs.model == 'ets':
         global_args = ets_args
         params_output_name = "ets-best-model.params"
         trials, epochs_per_trial  = 6, 50
    elif clargs.model == 'hierarchical':
         global_args = hierarchical_args
         params_output_name = "hierarchical-best-model.params"
         trials, epochs_per_trial  = 6, 50
    elif clargs.model == 'baseline':
         global_args = baseline_args
         params_output_name = "baseline-best-model.params"
         trials, epochs_per_trial  = 15, 20
    else:
         print("choose a valid model")
         sys.exit(0)

    # Before running go to trainer.py and uncomment line#12
    main(model_name=clargs.model, num_samples=trials, max_num_epochs=epochs_per_trial, gpus_per_trial=1, filename=params_output_name)
