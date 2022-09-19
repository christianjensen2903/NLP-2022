from gensim.models import Word2Vec as GensimWord2Vec
from models.Model import Model
import numpy as np


class Word2Vec(Model):
    def __init__(self):
        pass

    def extract_X(self, dataset):
        return dataset['tokenized_question'].to_list() + dataset['tokenized_plaintext'].to_list()

    def train(self, X):
        self.model = GensimWord2Vec(X, min_count=1)

    def predict(self, X):
        # Handle words missing from the vocabulary
        output = np.array([
            self.model.wv[word] for word in X if word in self.model.wv
        ])
        # If all words are missing, return a zero vector
        return output if len(output) != 0 else np.zeros((1, self.model.vector_size))

    def evaluate(self, X, y):
        pass

    def save(self, language):
        self.model.save(
            self.get_save_path(language.name, 'model')
        )

    def load(self, language):
        self.model = GensimWord2Vec.load(
            self.get_save_path(language.name, 'model')
        )
