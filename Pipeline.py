from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset
from Preprocess import Preprocess
from Language import Language

class Pipeline():
    def __init__(self, language: Language):
        self.language = language
        data = self.get_data()
        self.train_data, self.test_data = train_test_split(data, test_size=0.2)

    def get_data(self):
        """Get the preprocessed data"""
        destination = f'cleaned_data/{self.language.name}.csv'

        try:
            # Try to load the data from a file
            data = pd.read_csv(destination)
        except:
            # If the file doesn't exist, fetch the data from the internet and preprocess it
            data = load_dataset('copenlu/answerable_tydiqa')
            data = self.filter_language(data)
            data = Preprocess(data, self.language.tokenizer, self.language.cleaner).preprocess()
            data.to_csv(destination)

        return data
    
    def filter_language(self, dataset):
        """Filter the dataset to only contain the language of interest"""
        return dataset.filter(lambda x: x['language'] == self.language.name)

    def train(self, model):
        """Train the model"""
        self.Y = self.train_data['is_answerable']
        self.model = model.fit(self.X, self.Y)
        print(f'Training accuracy: {self.model.score(self.X, self.Y)}')

    def validate(self):
        """Validate the model"""
        # self.X_validation = self.vectorizer.transform(self.validation_data['tokenized_question'])
        self.Y_validation = self.validation_data['is_answerable']
        print(f'Validation accuracy: {self.model.score(self.X_validation, self.Y_validation)}')
        

    
