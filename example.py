from data_loading.datasets import DefaultDataset
from data_loading.dataloaders import get_data_loaders
from settings import ELL_DATA_DIR
from transformers import AutoTokenizer


if __name__ == "__main__":
    # instantiate the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
    # instantiate the dataset
    dataset = DefaultDataset(file_path=ELL_DATA_DIR, input_col='full_text', target_cols=['cohesion'], index_col='text_id', 
                             tokenizer=tokenizer)
                             
    # get the dataloaders. can make test and val sizes 0 if you don't want them
    train_dl, val_dl, test_dl = get_data_loaders(dataset, val_size=0.2, test_size=0.2, batch_size=32, val_batch_size=16,
    test_batch_size=16, num_workers=0)
    # iterate over the train dataloader
    for inputs, responses in train_dl:
        print(inputs)
        print(responses)
        break