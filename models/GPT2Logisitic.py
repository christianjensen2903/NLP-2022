from models.BOWLogistic import BOWLogistic
from models.GPT2Generator import GPT2Generator


class GPT2Logistic(BOWLogistic):
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
        self.GPT2Generator = self.get_GPT2Generator(dataset)
        return self.GPT2Generator.extract_X(dataset)
