from Model import Model
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

class BinaryQuestionClassifier(Model):
    def __init__(self):
        self.question_vectorizer = None
        self.plaintext_vectorizer = None

    def extract_X(self, dataset):
        """Extract features from the dataset"""
        
        # Extract bag of words for question and document
        if self.plaintext_vectorizer is None or self.plaintext_vectorizer is None:
            self.plaintext_vectorizer = CountVectorizer(tokenizer= lambda x: x,preprocessor=lambda x: x) # Avoid tokenizing again
            self.question_vectorizer = CountVectorizer(tokenizer= lambda x: x,preprocessor=lambda x: x)
            question_bow = self.question_vectorizer.fit_transform(dataset['tokenized_question'])
            plaintext_bow = self.plaintext_vectorizer.fit_transform(dataset['tokenized_plaintext'])
        else:
            question_bow = self.question_vectorizer.transform(dataset['tokenized_question'])
            plaintext_bow = self.plaintext_vectorizer.transform(dataset['tokenized_plaintext'])

        overlap = np.array([len([e for e in question if e in plaintext]) for question, plaintext in zip(dataset['cleaned_question'], dataset['cleaned_plaintext'])])

        # Concatenate the features
        X = np.concatenate((question_bow.toarray(), plaintext_bow.toarray(), overlap.reshape(-1,1)), axis=1)

        return X


    def train(self, X, y):
        """Train the model"""
        self.model = LogisticRegression(n_jobs=-1).fit(X, y)
        print(f'Training accuracy: {self.model.score(X, y)}')


    def predict(self, X):
        """Predict the answer"""
        pass

    def evaluate(self, X, y):
        """Evaluate the model"""
        print(f'Validation accuracy: {self.model.score(X, y)}')

    def save(self):
        """Save the model"""
        pass

    def load(self):
        """Load the model"""
        pass