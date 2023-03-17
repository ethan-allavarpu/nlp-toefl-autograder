from data_loading.datasets import DefaultDataset, CombinedDataset
from settings import *
import argparse

def get_argparser():
    argp = argparse.ArgumentParser()
    argp.add_argument('function', help="Choose pretrain, finetune, or evaluate") #TODO: add behavior for pretrain and eval
    argp.add_argument("--model_type", type=str, default="base", required=False)
    argp.add_argument("--val_losses_path", type=str, required=False)
    argp.add_argument('--writing_params_path', type=str, help='Path to the writing params file', required=False)
    argp.add_argument('--reading_params_path', type=str, help='Path to the reading params file', required=False)
    argp.add_argument('--loss_path', type=str, help='Path to the output losses', default="losses.txt", required=False)
    argp.add_argument('--outputs_path', type=str, help='Path to the output predictions', default="predictions.txt", required=False)
    argp.add_argument('--in_distribution_outputs_path', type=str, help='Path to the in-distribution output predictions', default="predictions.txt", required=False)
    argp.add_argument('--tokenizer_name', type=str, help='Name of the tokenizer to use', default="distilbert-base-uncased", required=False)
    argp.add_argument('--dataset', type=str, help='Name of the dataset to use', default="ICNALE-EDITED", required=False)
    argp.add_argument("--ICNALE_output", type=str, help="Use 'categories' or 'overall' score", default="overall", required=False)
    argp.add_argument('--max_epochs', type=int, help='Number of epochs to train for', default=20, required=False)
    argp.add_argument('--learning_rate', type=float, help='Learning rate', default=2e-5, required=False)
    argp.add_argument('--lr_decay', type=str, help='Decay Learning Rate', default="False", required=False)
    return argp

def get_dataset(args, tokenizer):
    if args.model_type != "hierarchical":
        if args.dataset == "ELL":
            return DefaultDataset(file_path=ELL_DATA_DIR, input_col='full_text', target_cols=['cohesion', 'syntax',  'vocabulary',  'phraseology',  'grammar',  'conventions'], index_col='text_id', 
                                    tokenizer=tokenizer)
        elif args.dataset == "ICNALE-EDITED":
            if args.ICNALE_output == "overall":
                return DefaultDataset(file_path=ICNALE_EDITED_DATA_DIR, input_col='essay', target_cols=['Total 1 (%)'], index_col=None, 
                                        tokenizer=tokenizer)
            else:
                return DefaultDataset(file_path=ICNALE_EDITED_DATA_DIR, input_col='essay', target_cols=['Content (/12)', 'Organization (/12)',
            'Vocabulary (/12)', 'Language Use (/12)', 'Mechanics (/12)'], index_col=None, 
                                        tokenizer=tokenizer)
        elif args.dataset == "ICNALE-WRITTEN": #TODO: lots of missing fields in this dataset, what imputations should we use?
            return DefaultDataset(file_path=ICNALE_WRITTEN_DATA_DIR, input_col='essay', target_cols=['Score'], index_col="sortkey", 
                                    tokenizer=tokenizer)
        elif args.dataset == "FCE":
            return DefaultDataset(file_path=FCE_DATA_DIR, input_col='essay', target_cols=['overall_score'], 
                                    tokenizer=tokenizer)
        elif args.dataset == "ETS":
            return DefaultDataset(file_path=ETS_DATA_DIR, input_col='response', target_cols=['low', 'medium', 'high'], 
                                    tokenizer=tokenizer, normalize=False)
        else:
            raise ValueError("Invalid dataset name")
    elif args.model_type == "hierarchical":
        if args.dataset == "ELL-ICNALE":
            return CombinedDataset(
                file_path1=ELL_DATA_DIR, file_path2=ICNALE_EDITED_DATA_DIR, input_col1="full_text", input_col2="essay",
                target_cols1=['cohesion', 'syntax',  'vocabulary',  'phraseology',  'grammar',  'conventions'],
                target_cols2=['Total 1 (%)'], tokenizer=tokenizer)
        elif args.dataset == "FCE":
            return DefaultDataset(file_path=FCE_DATA_DIR, input_col='essay', target_cols=['overall_score'], 
                                    tokenizer=tokenizer)
        elif args.dataset == "ICNALE-EDITED":
            if args.ICNALE_output == "overall":
                return DefaultDataset(file_path=ICNALE_EDITED_DATA_DIR, input_col='essay', target_cols=['Total 1 (%)'], index_col=None, 
                                        tokenizer=tokenizer)
            else:
                return DefaultDataset(file_path=ICNALE_EDITED_DATA_DIR, input_col='essay', target_cols=['Content (/12)', 'Organization (/12)',
            'Vocabulary (/12)', 'Language Use (/12)', 'Mechanics (/12)'], index_col=None, 
                                        tokenizer=tokenizer)
        elif args.dataset == "ETS":
            return DefaultDataset(file_path=ETS_DATA_DIR, input_col='response', target_cols=['score'], 
                                    tokenizer=tokenizer, normalize=False)
        else:
            raise ValueError("Invalid dataset name")

def write_predictions(path, predictions):
    with open(path, 'w') as f:
        for pred in predictions:
            f.write(f"{pred[0]},{pred[1]}\n")
