import imp
from models.feature_extraction.BOW import BOW
from models.XGBoost.XGBoost import XGBoost
from pickle import dump, load

class BOWXGBoost(XGBoost , BOW):
    def __init__(self, language):
        super().__init__(language)
