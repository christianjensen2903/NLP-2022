from models.feature_extraction.CBOW_BOW import CBOW_BOW
from models.Logistic.Logistic import Logistic

from gensim.models import Word2Vec as GensimWord2Vec
from models.Word2Vec import Word2Vec
import numpy as np


<<<<<<< Updated upstream
class CBOW_BOWLogistic(Logistic , CBOW_BOW):
    def __init__(self, language):
        super().__init__(language)
        self.word2vec = None
=======
class CBOW_BOWLogistic(Logistic, CBOW_BOW):
    def __init__(self):
        super().__init__()
        self.word2vec = None

    def weights(self):
        print("len", len(self.model.coef_[0]))
        print("other",(
              len(self.question_vectorizer.vocabulary_) +
              len(self.plaintext_vectorizer.vocabulary_) +
              len(self.first_word_vectorizer.vocabulary_) +
              1 +
              100 +
              100 +
              1 +
              1
        ))
        return dict(zip(
            list(map(lambda x: "*QUESTION* " + x, [*self.question_vectorizer.vocabulary_])) +
            list(map(lambda x: "*PLAINTEXT* " + x, [*self.plaintext_vectorizer.vocabulary_])) +
            list(map(lambda x: "*FIRSTWORD* " + x, [*self.first_word_vectorizer.vocabulary_])) +
            ['*OVERLAP*'] +
            list(map(lambda x: f'*QUESTION_CONTINOUS* {x}', range(100))) +
            list(map(lambda x: f'*PLAINTEXT_CONTINOUS* {x}', range(100))) +
            ['*EUCLIDEAN*'] +
            ['*COSINE*'],
            list(self.model.coef_[0]),
        ))

    def explainability(self, n=5):
        print(
            "EXPLAINABILITY:\n",
            "Top {} weights for positive:\n".format(n),
            sorted(self.weights().items(), key=lambda item: item[1], reverse=True)[
                :n],  # n most positive
            "\n\n",
            "Top {} weights for negative:\n".format(n),
            sorted(self.weights().items(), key=lambda item: item[1], reverse=False)[
                :n]  # n most negative
        )
>>>>>>> Stashed changes
