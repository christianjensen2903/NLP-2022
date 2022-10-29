from models.Logistic.Logistic import Logistic
from models.feature_extraction.CBOW import CBOW


class CBOWLogistic(Logistic , CBOW):
    def __init__(self):
        super().__init__()
