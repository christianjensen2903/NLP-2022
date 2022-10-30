from models.feature_extraction.feature_extracion import feature_extraction
from models.MultiGPT2Generator import MultiGPT2Generator
import numpy as np


class MultiGPT2Feature(feature_extraction):
    def __init__(self):
        super().__init__()

    def get_GPT2Generator(self, dataset):
        gpt2 = MultiGPT2Generator()
        gpt2.set_language(self.language)
        try:
            gpt2.load()
        except:
            gpt2.train(gpt2.extract_X(dataset), None)
            gpt2.save()
        return gpt2

    def extract_X(self, dataset):
        self.GPT2Generator = self.get_GPT2Generator(dataset)
        gpt2_output = self.GPT2Generator.predict(
            self.GPT2Generator.extract_X(dataset)
        )
        return gpt2_output
