from models.ContinuousBOWLogistic import ContinuousBOWLogistic
from models.GPT2Generator import GPT2Generator
import numpy as np


class GPT2ContinuousBOWLogistic(ContinuousBOWLogistic):
    def __init__(self):
        super().__init__()
        self.GPT2Generator = None

    def get_GPT2Generator(self, dataset):
        gpt2 = GPT2Generator()
        gpt2.set_language(self.language)
        try:
            gpt2.load()
        except:
            gpt2.train(gpt2.extract_X(dataset))
            gpt2.save()
        return gpt2

    def extract_X(self, dataset):
        bow_continous = super().extract_X(dataset)
        self.GPT2Generator = self.get_GPT2Generator(dataset)
        gpt2_output = self.GPT2Generator.predict(
            self.GPT2Generator.extract_X(dataset)
        )
        return np.concatenate(
            (
                bow_continous,
                gpt2_output
            ),
            axis=1
        )