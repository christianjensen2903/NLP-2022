from models.Logistic.Logistic import Logistic
from models.GPT2Generator import GPT2Generator
import numpy as np


class GPT2Logistic(Logistic):
    def __init__(self, language):
        super().__init__(language)
        self.GPT2Generator = None

    def get_GPT2Generator(self, dataset):
        gpt2 = GPT2Generator(self.langauge)
        gpt2.set_language(self.language)
        try:
            gpt2.load()
        except:
            gpt2.train(gpt2.extract_X(dataset))
            gpt2.save()
        return gpt2

    def extract_X(self, dataset):
        self.GPT2Generator = self.get_GPT2Generator(dataset)
        gpt2_output = self.GPT2Generator.predict(
            self.GPT2Generator.extract_X(dataset)
        )
        return gpt2_output
