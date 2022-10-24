from models.feature_extraction.BOW import BOW
from models.MLP.MLP import MLP
from pickle import dump, load


class BOWMLP(MLP , BOW):
    def __init__(self, language):
        super().__init__(language)