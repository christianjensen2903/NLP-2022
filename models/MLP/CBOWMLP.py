from models.MLP.MLP import MLP
from models.feature_extraction.CBOW import CBOW


class CBOWMLP(MLP , CBOW):
    def __init__(self):
        super().__init__()