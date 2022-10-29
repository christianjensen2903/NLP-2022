from sklearn.feature_extraction.text import CountVectorizer
from extractors.Extractor import Extractor
from pickle import dump, load
import numpy as np


def identity_function(x):
    return x


class BOW(Extractor):
    def __init__(self, language, dataset):
        super().__init__(language, dataset)

    def setup(self, train_data):
        self.plaintext_vectorizer = CountVectorizer(
            tokenizer=identity_function,
            preprocessor=identity_function
        ).fit(
            train_data['tokenized_plaintext']
        )
        self.question_vectorizer = CountVectorizer(
            tokenizer=identity_function,
            preprocessor=identity_function
        ).fit(
            train_data['tokenized_question']
        )
        self.first_word_vectorizer = CountVectorizer(
            tokenizer=identity_function,
            preprocessor=identity_function
        ).fit(
            train_data['tokenized_question'].str[0]
        )

    def run(self, data):
        question_bow = self.question_vectorizer.transform(
            data['tokenized_question']
        )
        plaintext_bow = self.plaintext_vectorizer.transform(
            data['tokenized_plaintext']
        )
        first_word_bow = self.first_word_vectorizer.transform(
            data['tokenized_question'].str[0]
        )
        overlap = np.array([
            len([e for e in question if e in plaintext])
            for question, plaintext in zip(data['cleaned_question'], data['cleaned_plaintext'])
        ])
        X = np.concatenate(
            (
                question_bow.toarray(),
                plaintext_bow.toarray(),
                first_word_bow.toarray(),
                overlap.reshape(-1,  1),
            ),
            axis=1
        )
        y = data['is_answerable']
        return X, y

    def save(self):
        dump(
            (
                self.plaintext_vectorizer,
                self.question_vectorizer,
                self.first_word_vectorizer
            ),
            open(self.get_save_path('pkl'), 'wb')
        )

    def load(self):
        self.plaintext_vectorizer, self.question_vectorizer, self.first_word_vectorizer = load(
            open(self.get_save_path('pkl'), 'rb')
        )
