from models.RandomForest.RandomForest import RandomForest
from models.feature_extraction.GPT2Feature import GPT2Feature


class GPT2RandomForest(RandomForest, GPT2Feature):
    def __init__(self):
        super().__init__()
