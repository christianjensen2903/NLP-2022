from typing import Callable
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class Pipeline():
    def __init__(self, dataset):
        self.train_data, self.validation_data = self.split_data(dataset)


    def split_data(self, dataset):
        """Split the dataset into train and validation sets."""
        return dataset["train"], dataset["validation"]


    def tokenize(self, tokenizer):
        """Tokenize the dataset"""
        def add_tokenization(row):
            row['tokenized_question'] = tokenizer(row['question_text'])
            row['tokenized_plaintext'] = tokenizer(row['document_plaintext'])
            return row

        self.train_data = self.train_data.map(add_tokenization)
        self.validation_data = self.validation_data.map(add_tokenization)

    def label_answerable(self):
        """Label the dataset"""
        def add_label(row):
            row['is_answerable'] = row["annotations"]['answer_start'] != [-1]
            return row
        self.train_data = self.train_data.map(add_label)
        self.validation_data = self.validation_data.map(add_label)


    def clean(self, cleaner):
        """Clean the dataset"""
        def clean_row(row):
            row['cleaned_question'] = cleaner(row['tokenized_question'])
            row['cleaned_plaintext'] = cleaner(row['tokenized_plaintext'])
            return row

        self.train_data = self.train_data.map(clean_row)
        self.validation_data = self.validation_data.map(clean_row)


    def train(self, model):
        """Train the model"""
        self.Y = self.train_data['is_answerable']
        self.model = model.fit(self.X, self.Y)
        print(f'Training accuracy: {self.model.score(self.X, self.Y)}')

    
    def validate(self):
        """Validate the model"""
        # self.X_validation = self.vectorizer.transform(self.validation_data['tokenized_question'])
        self.Y_validation = self.validation_data['is_answerable']
        print(f'Validation accuracy: {self.model.score(self.X_validation, self.Y_validation)}')
        

    
