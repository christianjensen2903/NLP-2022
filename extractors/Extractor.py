from abc import ABC
import os

class Extractor(ABC):
    def __init__(self, language, dataset):
        self.language = language
        try:
            self.load()
        except:
            self.setup(dataset)
            self.save()

    def get_save_path(self, filetype: str = ''):
        """Get the path of the extractor for loading and saving"""
        # Get the path of type saved_models/model_type
        path = os.path.join(
            'saved_extractors',
            self.__class__.__name__
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
        """Runs data dependent processes"""
        pass

    def run(self, data):
        """Extract X and y from the data"""
        pass

    def save(self):
        """Save the model"""
        pass

    def load(self):
        """Load the model"""
        raise NotImplementedError
