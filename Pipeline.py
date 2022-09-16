from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset
from Preprocess import Preprocess
from Model import Model


class Pipeline():
    def __init__(self):
        pass

    def get_data(self, language, preproccesor: Preprocess):
        """Get the preprocessed data"""
        destination = f'cleaned_data/{language}.json'

        print('\nLoading data...')
        try:
            # Try to load the data from a file
            data = pd.read_json(destination, orient='records')
        except:
            # If the file doesn't exist, fetch the data from the internet and preprocess it
            data = load_dataset('copenlu/answerable_tydiqa')
            print('\nPreproccessing the data...')
            data = self.filter_language(data, language)
            data = preproccesor.preprocess(data)
            data.to_json(destination, orient='records')

        return data

    def filter_language(self, dataset, language):
        """Filter the dataset to only contain the language of interest"""
        return dataset.filter(lambda x: x['language'] == language)

    def split_data(self, data):
        """Split the data into training and validation data"""
        # Random state is needed for saving and loading the model
        train_data, validation_data = train_test_split(
            data, 
            test_size=0.2, 
            random_state=0
        )
        return train_data, validation_data

    def train(self, model: Model, X, y):
        """Train the model"""
        print('\nTraining the model...')
        model.train(X, y)
        print(f'Train score: {model.evaluate(X, y)}')
        return model

    def evaluate(self, model: Model, X, y):
        """Evaluate the model"""
        print('\nEvaluating the model...')
        print(f'Validation score: {model.evaluate(X, y)}')
