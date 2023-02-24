import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from modeling.model import SpeechModel
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoFeatureExtractor
import random
import argparse

from modeling import trainer
from data_loading.datasets import SpeechDataset
from data_loading.dataloaders import get_data_loaders
from settings import SPEECHOCEAN_DATA_DIR


torch.manual_seed(0)
argp = argparse.ArgumentParser()
argp.add_argument('function', help="Choose pretrain, finetune, or evaluate") #TODO: add behavior for pretrain and eval
argp.add_argument('--writing_params_path', type=str, help='Path to the writing params file', required=False)
argp.add_argument('--reading_params_path', type=str, help='Path to the reading params file', required=False)
argp.add_argument('--outputs_path', type=str, help='Path to the output predictions', default="predictions.txt", required=False)
argp.add_argument('--tokenizer_name', type=str, help='Name of the tokenizer to use', default="facebook/wav2vec2-base", required=False)
argp.add_argument('--dataset', type=str, help='Name of the dataset to use', default="ICNALE-EDITED", required=False)
argp.add_argument('--max_epochs', type=int, help='Number of epochs to train for', default=50, required=False)
argp.add_argument('--learning_rate', type=float, help='Learning rate', default=2e-5, required=False)
args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'


# instantiate the tokenizer
tokenizer = AutoFeatureExtractor.from_pretrained(args.tokenizer_name)
# instantiate the dataset
if args.dataset == "SPEECHOCEAN":
    dataset = SpeechDataset(path_name=SPEECHOCEAN_DATA_DIR, input_col = 'audio', target_cols=['accuracy', 'completeness', 'fluency', 'prosodic', 'total'], tokenizer=tokenizer)
else:
    raise ValueError("Invalid dataset name")
                             

if args.function == 'pretrain':
    pass

elif args.function == 'finetune':
    # get the dataloaders. can make test and val sizes 0 if you don't want them
    train_dl, val_dl, test_dl = get_data_loaders(dataset, val_size=0, test_size=0.2, batch_size=16, val_batch_size=1,
        test_batch_size=1, num_workers=0)
    # TensorBoard training log
    writer = SummaryWriter(log_dir='expt/')

    train_config = trainer.TrainerConfig(max_epochs=args.max_epochs, 
            learning_rate=args.learning_rate, 
            num_workers=4, writer=writer, ckpt_path='expt/params.pt')

    model = SpeechModel(num_outputs=len(dataset.targets.columns), pretrain_model_name=args.tokenizer_name)
    trainer = trainer.Trainer(model, train_dl, test_dl, train_config)
    trainer.train()
    torch.save(model.state_dict(), args.writing_params_path)

elif args.function == 'evaluate':
    train_dl, val_dl, test_dl = get_data_loaders(dataset, val_size=0, test_size=0.2, batch_size=16, val_batch_size=1,
        test_batch_size=1, num_workers=0)
    model = SpeechModel(num_outputs=len(dataset.targets.columns), pretrain_model_name=args.tokenizer_name)

    model.load_state_dict(torch.load(args.reading_params_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()
    predictions = []

    pbar = tqdm(enumerate(test_dl), total=len(test_dl)) 
    for it, (x, y) in pbar:
        # place data on the correct device
        x = x.to(device)
        predictions.append((dict(zip(dataset.targets.columns, model(x)[0][0].tolist())),
        dict(zip(dataset.targets.columns, y[0].tolist()))))
        torch.cuda.empty_cache()

    with open(args.outputs_path, 'w') as f:
        for pred in predictions:
            f.write(f"{pred[0]},{pred[1]}\n")
    

else:
    print("Invalid function name. Choose pretrain, finetune, or evaluate")
                             