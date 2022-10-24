from models.feature_extraction.BOW import BOW
from models.RandomForest.RandomForest import RandomForest
from pickle import dump, load


class BOWRandomForest(RandomForest , BOW):
    def __init__(self, language):
        super().__init__(language)