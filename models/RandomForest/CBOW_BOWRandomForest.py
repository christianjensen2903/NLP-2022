from models.feature_extraction.CBOW_BOW import CBOW_BOW
from models.RandomForest.RandomForest import RandomForest

from gensim.models import Word2Vec as GensimWord2Vec
from models.Word2Vec import Word2Vec
import numpy as np


class CBOW_BOWRandomForest(RandomForest , CBOW_BOW):
    def __init__(self):
        super().__init__()
        self.word2vec = None