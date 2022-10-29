from models.XGBoost.XGBoost import XGBoost
from models.feature_extraction.GPT2Feature import GPT2Feature


class GPT2XGBoost(XGBoost, GPT2Feature):
    def __init__(self):
        super().__init__()
