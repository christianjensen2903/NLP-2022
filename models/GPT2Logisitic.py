from sklearn.linear_model import LogisticRegression
from models.BOWLogistic import BOWLogistic


class GPT2Logistic(BOWLogistic):
    def __init__(self):
        super().__init__()

    def extract_X(self, dataset):
        pass
