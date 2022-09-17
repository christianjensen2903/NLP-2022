from models.BOWLogistic import BOWLogistic
from gensim.models import Word2Vec as GensimWord2Vec
from models.Word2Vec import Word2Vec
import numpy as np


class ContinuousBOWLogistic(BOWLogistic):
    def __init__(self):
        super().__init__()
        self.word2vec = None

    def get_word2vec(self, dataset, language):
        word2vec = Word2Vec()
        try:
            word2vec.load(language)
        except:
            word2vec.train(word2vec.extract_X(dataset))
            word2vec.save(language)
        return word2vec

    def get_continuous_representation(self, dataset, language):
        if self.word2vec is None:
            self.word2vec = self.get_word2vec(dataset, language)
        question_mean_representation = np.array([
            self.word2vec.predict(sentence).mean(axis=0) for sentence in dataset['tokenized_question']
        ])
        plaintext_mean_representation = np.array([
            self.word2vec.predict(sentence).mean(axis=0) for sentence in dataset['tokenized_plaintext']
        ])
        return np.concatenate(
            (
                question_mean_representation,
                plaintext_mean_representation
            ),
            axis=1
        )

    def extract_X(self, dataset, language):
        bow = super().extract_X(dataset, language)
        continuous_representation = self.get_continuous_representation(
            dataset,
            language
        )
        return np.concatenate(
            (
                bow,
                continuous_representation
            ),
            axis=1
        )
