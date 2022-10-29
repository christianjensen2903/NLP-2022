from gensim.models import Word2Vec as GensimWord2Vec
from models.Model import Model
import numpy as np


class Word2Vec(Model):
    def __init__(self, extractor, language, config={}):
        super().__init__(extractor, language, config)

    def train(self, X):
        self.model = GensimWord2Vec(X, min_count=1)

    def predict(self, X):
        # Handle words missing from the vocabulary
        return np.array([
            self.model.wv[word] if word in self.model.wv else np.zeros(
                self.model.vector_size)
            for word in X
        ])

    def save(self):
        self.model.save(self.get_save_path('model'))

    def load(self):
        self.model = GensimWord2Vec.load(self.get_save_path('model'))
