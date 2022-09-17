from abc import ABC, abstractmethod
import os


class Model(ABC):

    def get_save_path(self, language: str, filetype: str):
        """Get the path of the model for loading and saving"""
        # Get the path of type saved_models/model_type
        path = os.path.join(
            'saved_models',
            self.__class__.__name__
        )
        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Return the path of type saved_models/model_type/language.filetype
        return os.path.join(
            path,
            f'{language}.{filetype}'
        )

    def extract_X(self, dataset, language: str = ""):
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

    def save(self, language: str):
        """Save the model"""
        pass

    def load(self, language: str):
        """Load the model"""
        pass
