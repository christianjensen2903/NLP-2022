from models.XGBoost.XGBoost import XGBoost
from models.feature_extraction.CBOW_BOW import CBOW_BOW


from models.XGBoost.XGBoost import XGBoost


class CBOW_BOWXGBoost(XGBoost , CBOW_BOW):
    def __init__(self):
        super().__init__()
        self.word2vec = None