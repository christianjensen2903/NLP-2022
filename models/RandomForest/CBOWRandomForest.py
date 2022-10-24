from models.RandomForest.RandomForest import RandomForest
from models.feature_extraction.CBOW import CBOW


class CBOWRandomForest(RandomForest , CBOW):
    def __init__(self, language):
        super().__init__(language)