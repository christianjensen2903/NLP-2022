from abc import ABC, abstractmethod

class feature_extraction(ABC):
    def set_language(self, language: str):
        self.language = language

    def extract_X(self, dataset):
        """Extract features from the dataset"""
        pass