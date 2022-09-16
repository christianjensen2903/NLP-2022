from abc import ABC, abstractmethod

class Model(ABC):

    def get_path(self, language):
        """Get the path of the model for loading and saving"""
        pass

    def extract_X(self, dataset, language):
        """Extract features from the dataset"""
        pass

    def train(self, X, y):
        """Train the model"""
        pass

    def predict(self, X):
        """Predict the answer"""
        pass

    def evaluate(self, X, y):
        """Evaluate the model"""
        pass

    def save(self, language):
        """Save the model"""
        pass

    def load(self, language):
        """Load the model"""
        pass
