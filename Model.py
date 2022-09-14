from abc import ABC, abstractmethod

class Model(ABC):
    
    def train(self):
        """Train the model"""
        pass

    def predict(self, X):
        """Predict the answer"""
        pass

    def evaluate(self):
        """Evaluate the model"""
        pass

    def save(self):
        """Save the model"""
        pass

    def load(self):
        """Load the model"""
        pass
