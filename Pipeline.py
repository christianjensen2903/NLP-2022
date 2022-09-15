from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset
from Preprocess import Preprocess

class Pipeline():
    def __init__(self):
        pass

    def get_data(self, language, preproccesor):
        """Get the preprocessed data"""
        destination = f'cleaned_data/{language}.json'

        print('\nLoading data...')
        try:
            # Try to load the data from a file
            data = pd.read_json(destination, orient='records')
        except:
            # If the file doesn't exist, fetch the data from the internet and preprocess it
            data = load_dataset('copenlu/answerable_tydiqa')
            data = self.filter_language(data, language)
            print('\nPreproccessing the data...')
            data = preproccesor.preprocess(data)
            data.to_json(destination, orient='records')

        return data
    
    def filter_language(self, dataset, language):
        """Filter the dataset to only contain the language of interest"""
        return dataset.filter(lambda x: x['language'] == language)

    def split_data(self, data):
        """Split the data into training and validation data"""
        train_data, validation_data = train_test_split(data, test_size=0.2)
        return train_data, validation_data

    def train(self, model):
        """Train the model"""
        print('\nTraining the model...')
        self.Y = self.train_data['is_answerable']
        self.model = model.fit(self.X, self.Y)
        print(f'Training accuracy: {self.model.score(self.X, self.Y)}')

    def validate(self):
        """Validate the model"""
        print('\nValidating the model...')
        # self.X_validation = self.vectorizer.transform(self.validation_data['tokenized_question'])
        self.Y_validation = self.validation_data['is_answerable']
        print(f'Validation accuracy: {self.model.score(self.X_validation, self.Y_validation)}')
        

    
