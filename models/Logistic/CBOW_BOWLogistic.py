from models.feature_extraction.CBOW_BOW import CBOW_BOW
from models.Logistic.Logistic import Logistic

from gensim.models import Word2Vec as GensimWord2Vec
from models.Word2Vec import Word2Vec
import numpy as np


class CBOW_BOWLogistic(Logistic , CBOW_BOW):
    def __init__(self, language):
        super().__init__(language)
        self.word2vec = None