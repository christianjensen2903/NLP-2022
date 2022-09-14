from Pipeline import Pipeline
import pandas as pd
from datasets import load_dataset
from Preprocess import Preprocess

class Language():
    def __init__(self, name, tokenizer, cleaner):
        self.name = name
        self.tokenizer = tokenizer
        self.cleaner = cleaner
        self.get_data()
        self.pipeline = Pipeline(self.data)

    def get_data(self):
        """Get the preprocessed data"""
        destination = f'cleaned_data/{self.name}.csv'

        try:
            # Try to load the data from a file
            self.data = pd.read_csv(destination)
        except:
            # If the file doesn't exist, fetch the data from the internet and preprocess it
            self.data = load_dataset('copenlu/answerable_tydiqa')
            self.data = self.filter_language(self.data)
            self.data = Preprocess(self.data, self.tokenizer, self.cleaner).preprocess()
            self.data.to_csv(destination)
    

    def filter_language(self, dataset):
        """Filter the dataset to only contain the language of interest"""
        return dataset.filter(lambda x: x['language'] == self.name)
        # return dataset[dataset["language"] == self.name] 
    
    



    

