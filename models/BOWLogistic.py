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
        self.first_word_vectorizer = None
        self.model = LogisticRegression()

    def set_language(self, language):
        super().set_language(language)
        self.question_vectorizer = None
        self.plaintext_vectorizer = None
        self.first_word_vectorizer = None

    def extract_X(self, dataset):
        # Extract bag of words for question and document
        if self.plaintext_vectorizer is None or self.plaintext_vectorizer is None or self.first_word_vectorizer is None:
            self.plaintext_vectorizer = CountVectorizer(
                tokenizer=lambda x: x,  # Avoid tokenizing again
                preprocessor=lambda x: x
            )
            self.question_vectorizer = CountVectorizer(
                tokenizer=lambda x: x,
                preprocessor=lambda x: x
            )
            self.first_word_vectorizer = CountVectorizer(
                tokenizer=lambda x: x,
                preprocessor=lambda x: x
            )

            question_bow = self.question_vectorizer.fit_transform(
                dataset['tokenized_question']
            )
            plaintext_bow = self.plaintext_vectorizer.fit_transform(
                dataset['tokenized_plaintext']
            )
            first_word_bow = self.first_word_vectorizer.fit_transform(
                dataset['tokenized_question'].str[0]
            )
        else:
            question_bow = self.question_vectorizer.transform(
                dataset['tokenized_question']
            )
            plaintext_bow = self.plaintext_vectorizer.transform(
                dataset['tokenized_plaintext']
            )
            first_word_bow = self.first_word_vectorizer.transform(
                dataset['tokenized_question'].str[0]
            )

        overlap = np.array([len([e for e in question if e in plaintext]) for question, plaintext in zip(
            dataset['cleaned_question'], dataset['cleaned_plaintext'])])

        # Concatenate the features
        X = np.concatenate(
            (
                question_bow.toarray(),
                plaintext_bow.toarray(),
                first_word_bow.toarray(),  # First word in question
                overlap.reshape(-1,  1),  # Amount of words overlapping
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

    def save(self):
        dump(self.model, open(self.get_save_path('pkl'), 'wb'))

    def load(self):
        self.model = load(open(self.get_save_path('pkl'), 'rb'))

    def explainability(self):
        word_weigths = dict(zip(
            list(map(lambda x: "*QUESTION* " + x, [*self.question_vectorizer.vocabulary_])) +
            list(map(lambda x: "*PLAINTEXT* " + x, [*self.plaintext_vectorizer.vocabulary_])) +
            list(map(lambda x: "*FIRSTWORD* " + x, [*self.first_word_vectorizer.vocabulary_])) +
            ['*OVERLAP*'],
            list(self.model.coef_[0]),
        ))
        return (sorted(word_weigths.items(), key=lambda item: item[1], reverse=True)[:5], sorted(word_weigths.items(), key=lambda item: item[1], reverse=False)[:5])
