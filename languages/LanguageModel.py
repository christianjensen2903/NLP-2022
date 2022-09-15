from abc import ABC, abstractmethod

class LanguageModel():
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def clean(self, text):
        pass
    
    def tokenize(self, text):
        pass
    
    



    

