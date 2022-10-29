from abc import ABC
import os
import torch


class Model(ABC):
    def __init__(self, extractor, language: str = "english", config: dict = {}):
        self.name = self.__class__.__name__ + extractor.__name__
        self.extractor = extractor
        self.language = language.lower()
        self.config = config

        # Set the device to use
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def get_save_path(self, filetype: str = ''):
        """Get the path of the model for loading and saving"""
        # Get the path of type saved_models/model_type
        path = os.path.join(
            'saved_models',
            self.name
        )
        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Return the path of type saved_models/model_type/language.filetype
        # Or saved_models/model_type/language if the saving is done on a directory
        return os.path.join(
            path,
            f'{self.language}.{filetype}' if filetype else self.language
        )

    def setup(self, train_data):
        """Runs data dependent processes needed for training"""
        # Initialize extractor
        self.extractor = self.extractor(
            self.language, train_data
        )

    def extract(self, data):
        """Extracts X and y from the data"""
        return self.extractor.run(data)

    def train(self, X, y):
        """Train the model"""
        pass

    def predict(self, X):
        """Predict the answer"""
        pass

    def evaluate(self, X, y):
        """Evaluate the model"""
        pass

    def save(self):
        """Save the model"""
        pass

    def load(self):
        """Load the model"""
        raise NotImplementedError
