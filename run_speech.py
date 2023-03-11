import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from modeling.model import SpeechModel, SiameseSpeechModel
from modeling.model import SpeechModel, SiameseSpeechModel
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoFeatureExtractor
import random
import argparse

from modeling import trainer
from data_loading.datasets import SpeechDataset
from data_loading.dataloaders import split_on_indices
from settings import SPEECHOCEAN_DATA_DIR


torch.manual_seed(0)
argp = argparse.ArgumentParser()
argp.add_argument('function', help="Choose pretrain, finetune, or evaluate") #TODO: add behavior for pretrain and eval
argp.add_argument('--writing_params_path', type=str, help='Path to the writing params file', required=False)
argp.add_argument('--reading_params_path', type=str, help='Path to the reading params file', required=False)
argp.add_argument('--outputs_path', type=str, help='Path to the output predictions', default="predictions.txt", required=False)
argp.add_argument('--loss_path', type=str, help='Path to the output losses', default="losses.txt", required=False)
argp.add_argument('--tokenizer_name', type=str, help='Name of the tokenizer to use', default="facebook/wav2vec2-base", required=False)
argp.add_argument('--dataset', type=str, help='Name of the dataset to use', default="SPEECHOCEAN", required=False)
argp.add_argument('--max_epochs', type=int, help='Number of epochs to train for', default=25, required=False)
argp.add_argument('--learning_rate', type=float, help='Learning rate', default=2e-5, required=False)
args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'


# instantiate the tokenizers
tokenizer = AutoFeatureExtractor.from_pretrained(args.tokenizer_name)
# instantiate the dataset
if args.dataset == "SPEECHOCEAN":
    dataset = SpeechDataset(path_name=SPEECHOCEAN_DATA_DIR, input_col = 'audio', target_cols_sentence=['accuracy', 'fluency', 'prosodic', 'total'],
    target_cols_words = ["accuracy", "stress", "total"], target_cols_phones = ["phones-accuracy"], tokenizer=tokenizer)
else:
    raise ValueError("Invalid dataset name")
                             

if args.function == 'pretrain':
    pass

elif args.function == 'finetune':
    # get the dataloaders. can make test and val sizes 0 if you don't want them
    train_dl, val_dl, test_dl = split_on_indices(
        dataset,
        index_col="speaker_id",
        val_size=0.1,
        test_size=0.1,
        batch_size=tune_config["batch_size"],
        val_batch_size=16,
        test_batch_size=1,
        num_workers=0,
    )
    # TensorBoard training log
    writer = SummaryWriter(log_dir='expt/')

    train_config = trainer.TrainerConfig(max_epochs=args.max_epochs, 
            learning_rate=args.learning_rate, 
            num_workers=4, writer=writer, ckpt_path='expt/params.pt')

    model = SiameseSpeechModel(num_outputs=len(dataset.targets_sentence.columns), pretrain_model_name=args.tokenizer_name,
    phoneme_seq_length=dataset.phoneme_seq_length, word_seq_length=dataset.word_seq_length, word_outputs = 0 if dataset.targets_words is None else len(dataset.targets_words.columns))
    trainer = trainer.Trainer(model=model,  train_dataloader=train_dl, test_dataloader=test_dl, config=train_config, val_dataloader=val_dl)
    trainer.train(split='train', step=0)
    torch.save(model.state_dict(), args.writing_params_path)
    with open(args.loss_path, 'w') as f:
        for loss in trainer.losses:
            f.write(f"{loss[0]},{loss[1]}\n")

elif args.function == 'evaluate':
    train_dl, val_dl, test_dl = split_on_indices(
        dataset,
        index_col="speaker_id",
        val_size=0.1,
        test_size=0.1,
        batch_size=tune_config["batch_size"],
        val_batch_size=16,
        test_batch_size=1,
        num_workers=0,
    )
    model = SiameseSpeechModel(num_outputs=len(dataset.targets_sentence.columns), pretrain_model_name=args.tokenizer_name,
    phoneme_seq_length=dataset.phoneme_seq_length, word_seq_length=dataset.word_seq_length, word_outputs = len(dataset.targets_words.columns))

    model.load_state_dict(torch.load(args.reading_params_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()
    predictions = []

    pbar = tqdm(enumerate(test_dl), total=len(test_dl)) 
    # pred_cols = [f'pred_{c}' for c in dataset.targets_sentence.columns] + [f'pred_word_{c}' for c in dataset.targets_words.columns] + [f'pred_{c}' for c in dataset.targets_phones.columns]
    pred_cols = [f'pred_{c}' for c in dataset.targets_sentence.columns]
    actual_cols = [f'actual_{c}' for c in dataset.targets_sentence.columns]
    for it, (x, y) in pbar:
        # place data on the correct device
        x = x.to(device)
        one_output = (len(y)<3)
        predictions.append(({**dict(zip(pred_cols, model(x, one_output=one_output)[0][0][0].tolist())), **dict(zip(actual_cols, y[0][0].tolist()))}))
        torch.cuda.empty_cache()

    pd.DataFrame(predictions).to_csv(args.outputs_path, index=False)
    

else:
    print("Invalid function name. Choose pretrain, finetune, or evaluate")
                             