from models.MLP.MLP import MLP
from models.feature_extraction.GPT2Feature import GPT2Feature


class GPT2MLP(MLP, GPT2Feature):
    def __init__(self):
        super().__init__()
