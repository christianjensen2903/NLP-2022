from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from pickle import dump, load
from models.feature_extraction.feature_extracion import feature_extraction
import numpy as np


class BOW(feature_extraction):
    def __init__(self):
        super().__init__()
        self.question_vectorizer = None
        self.plaintext_vectorizer = None
        self.first_word_vectorizer = None
        # self.model = LogisticRegression()

    def set_language(self, language):
        super().set_language(language)
        self.question_vectorizer = None
        self.plaintext_vectorizer = None
        self.first_word_vectorizer = None

    def identity_function(self , x):
        return x

    def extract_X(self, dataset):
        # Extract bag of words for question and document
        if self.plaintext_vectorizer is None or self.plaintext_vectorizer is None or self.first_word_vectorizer is None:
            self.plaintext_vectorizer = CountVectorizer(
                tokenizer=self.identity_function,  # Avoid tokenizing again
                preprocessor=self.identity_function
            )
            self.question_vectorizer = CountVectorizer(
                tokenizer=self.identity_function,
                preprocessor=self.identity_function
            )
            self.first_word_vectorizer = CountVectorizer(
                tokenizer=self.identity_function,
                preprocessor=self.identity_function
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