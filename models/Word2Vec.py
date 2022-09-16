from gensim.models import Word2Vec as GensimWord2Vec
from models.Model import Model
import numpy as np

class Word2Vec(Model):
    def __init__(self):
        pass

    def get_path(self, language):
        return f'saved_models/word2vec[{language.name}].model'

    def extract_X(self, dataset):
        return dataset['tokenized_question'].to_list() + dataset['tokenized_plaintext'].to_list()

    def train(self, X):
        self.model = GensimWord2Vec(X, min_count=1)

    def predict(self, X):
        # Handle words missing from the vocabulary
        return np.array([self.model.wv[word] for word in X if word in self.model.wv])

    def evaluate(self, X, y):
        pass

    def save(self, language):
        self.model.save(self.get_path(language))

    def load(self, language):
        self.model = GensimWord2Vec.load(self.get_path(language))
