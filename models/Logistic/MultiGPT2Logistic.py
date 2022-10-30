from models.Logistic.Logistic import Logistic
from models.feature_extraction.MultiGPT2Feature import MultiGPT2Feature


class MultiGPT2Logistic(Logistic, MultiGPT2Feature):
    def __init__(self):
        super().__init__()