from models.XGBoost.XGBoost import XGBoost
from models.feature_extraction.CBOW import CBOW


class CBOWXGBoost(XGBoost , CBOW):
    def __init__(self, language):
        super().__init__(language)
    
    def explainability(self):
        return