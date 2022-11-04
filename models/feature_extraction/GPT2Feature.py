<<<<<<<< HEAD:models/GPT2Logistic.py
from models.Logistic.Logistic import Logistic
from models.GPT2Generator import GPT2Generator
========
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from pickle import dump, load
from models.feature_extraction.feature_extracion import feature_extraction
>>>>>>>> before_change_to_model:models/feature_extraction/GPT2Feature.py
import numpy as np
from models.GPT2Generator import GPT2Generator


<<<<<<<< HEAD:models/GPT2Logistic.py
class GPT2Logistic(Logistic):
    def __init__(self, language):
        super().__init__(language)
        self.GPT2Generator = None
========
class GPT2Feature(feature_extraction):
    def __init__(self):
        super().__init__()
>>>>>>>> before_change_to_model:models/feature_extraction/GPT2Feature.py

    def get_GPT2Generator(self, dataset):
        gpt2 = GPT2Generator(self.langauge)
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
