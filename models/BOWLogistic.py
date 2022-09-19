from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from pickle import dump, load
from models.Model import Model
import numpy as np


class BOWLogistic(Model):
    def __init__(self):
        self.question_vectorizer = None
        self.plaintext_vectorizer = None
        self.model = LogisticRegression()

    def extract_X(self, dataset, language):
        # Extract bag of words for question and document
        if self.plaintext_vectorizer is None or self.plaintext_vectorizer is None:
            self.plaintext_vectorizer = CountVectorizer(
                tokenizer=lambda x: x,  # Avoid tokenizing again
                preprocessor=lambda x: x
            )
            self.question_vectorizer = CountVectorizer(
                tokenizer=lambda x: x,
                preprocessor=lambda x: x
            )
            question_bow = self.question_vectorizer.fit_transform(
                dataset['tokenized_question']
            )
            plaintext_bow = self.plaintext_vectorizer.fit_transform(
                dataset['tokenized_plaintext']
            )
        else:
            question_bow = self.question_vectorizer.transform(
                dataset['tokenized_question']
            )
            plaintext_bow = self.plaintext_vectorizer.transform(
                dataset['tokenized_plaintext']
            )

        overlap = np.array([len([e for e in question if e in plaintext]) for question, plaintext in zip(
            dataset['cleaned_question'], dataset['cleaned_plaintext'])])

        # Concatenate the features
        X = np.concatenate(
            (
                question_bow.toarray(),
                plaintext_bow.toarray(),
                overlap.reshape(-1,  1)
            ),
            axis=1
        )

        return X

    def train(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        return self.model.score(X, y)

    def save(self, language):
        dump(self.model, open(self.get_save_path(language, 'pkl'), 'wb'))

    def load(self, language):
        self.model = load(open(self.get_save_path(language, 'pkl'), 'rb'))
