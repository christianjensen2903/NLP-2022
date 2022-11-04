from models.feature_extraction.BOW import BOW
from models.Logistic.Logistic import Logistic
from sklearn.linear_model import LogisticRegression
from pickle import dump, load


class BOWLogistic(Logistic , BOW):
    def __init__(self):
        super().__init__()