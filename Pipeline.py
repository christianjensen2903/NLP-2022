from typing import Callable
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class Pipeline():
    def __init__(self, dataset):
        self.train_data, self.validation_data = self.split_data(dataset)
        self.explode_annotations()


    def split_data(self, dataset):
        """Split the dataset into train and validation sets."""
        return dataset["train"].to_pandas(), dataset["validation"].to_pandas()

    def explode_annotations(self):
        self.train_data['answer_start'] = self.train_data['annotations'].apply(lambda x: x['answer_start'][0])
        self.train_data['answer_text'] = self.train_data['annotations'].apply(lambda x: x['answer_text'][0])  
        self.validation_data['answer_start'] = self.validation_data['annotations'].apply(lambda x: x['answer_start'][0])
        self.validation_data['answer_text'] = self.validation_data['annotations'].apply(lambda x: x['answer_text'][0])
        self.train_data.drop(columns=['annotations'], inplace=True)
        self.validation_data.drop(columns=['annotations'], inplace=True)


    def tokenize(self, tokenizer):
        """Tokenize the dataset"""
        self.train_data['tokenized_question'] = self.train_data['question_text'].apply(tokenizer)
        self.validation_data['tokenized_question'] = self.validation_data['question_text'].apply(tokenizer)

        self.train_data['tokenized_plaintext'] = self.train_data['document_plaintext'].apply(tokenizer)
        self.validation_data['tokenized_plaintext'] = self.validation_data['document_plaintext'].apply(tokenizer)

    def label_answerable(self):
        """Label the dataset"""
        self.train_data['is_answerable'] = self.train_data['answer_start'].apply(lambda x: x != -1)
        self.validation_data['is_answerable'] = self.validation_data['answer_start'].apply(lambda x: x != -1)


    def balance(self):
        """Balance the dataset"""
        def balance_individual(data):
            answerable = data[data['is_answerable']]
            not_answerable = data[~data['is_answerable']]
            answerable = answerable.sample(n=not_answerable.shape[0], replace=True)
            return answerable.append(not_answerable)


    def clean(self, cleaner):
        """Clean the dataset"""
        self.train_data['cleaned_question'] = self.train_data['tokenized_question'].apply(cleaner)
        self.validation_data['cleaned_question'] = self.validation_data['tokenized_question'].apply(cleaner)

        self.train_data['cleaned_plaintext'] = self.train_data['tokenized_plaintext'].apply(cleaner)
        self.validation_data['cleaned_plaintext'] = self.validation_data['tokenized_plaintext'].apply(cleaner)


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
        

    
