from models.Logistic.Logistic import Logistic
from models.feature_extraction.GPT2Feature import GPT2Feature


class GPT2Logistic(Logistic, GPT2Feature):
    def __init__(self):
        super().__init__()
