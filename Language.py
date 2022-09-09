from typing import Callable
from Pipeline import Pipeline

class Language():
    def __init__(self, name: str, tokenizer: Callable[[str], list[str]],cleaner: Callable[[str], str], pipeline: Pipeline):
        self.name = name
        self.tokenizer = tokenizer
        self.cleaner = cleaner
        self.pipeline = pipeline


    

