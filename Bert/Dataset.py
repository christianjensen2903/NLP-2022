from transformers import BertTokenizer
import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.X = [
            self.tokenizer(
                text,
                padding='max_length',
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ) for text in X
        ]
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
