from typing import Callable
from sklearn.feature_extraction.text import CountVectorizer


class Pipeline():
    def __init__(self, dataset):
        self.train, self.validation = self.split_data(dataset)


    def split_data(self, dataset):
        """Split the dataset into train and validation sets."""
        return dataset["train"], dataset["validation"]


    def tokenize(self, tokenizer):
        """Tokenize the dataset"""
        def add_tokenization(row):
            row['tokenized_question'] = tokenizer(row['question_text'])
            return row

        self.train = self.train.map(add_tokenization)
        self.validation = self.validation.map(add_tokenization)

    def label_answerable(self):
        """Label the dataset"""
        def add_label(row):
            row['is_answerable'] = int(len(row["annotations"]['answer_start']) != 0)
            return row
        self.train = self.train.map(add_label)
        self.validation = self.validation.map(add_label)


    def extract_features(self):
        """Extract features from the dataset"""
        # Avoid tokenizing again
        self.vectorizer = CountVectorizer(
                tokenizer= lambda x: x,
                preprocessor=lambda x: x,
        )
        self.X = self.vectorizer.fit_transform(self.train['tokenized_question'])


    def train(self, model):
        """Train the model"""
        self.Y = self.train['is_answerable']
        self.model = model.fit(self.X, self.Y)
        print(f'Training accuracy: {self.model.score(self.X, self.Y)}')

    
    def validate(self):
        """Validate the model"""
        self.X_validation = self.vectorizer.transform(self.validation['tokenized_question'])
        self.Y_validation = self.validation['is_answerable']
        print(f'Validation accuracy: {self.model.score(self.X_validation, self.Y_validation)}')
        

    