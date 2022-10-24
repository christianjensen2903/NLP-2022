from models.MLP.MLP import MLP
from models.feature_extraction.CBOW import CBOW


class CBOWMLP(MLP , CBOW):
    def __init__(self, language):
        super().__init__(language)