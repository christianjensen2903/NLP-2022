from Pipeline import Pipeline
import pandas as pd
from datasets import load_dataset

class Language():
    def __init__(self, name, tokenizer, cleaner):
        self.name = name
        self.tokenizer = tokenizer
        self.cleaner = cleaner
        self.get_data()
        self.pipeline = Pipeline(self.data)

    def get_data(self):
        destination = f'cleaned_data/{self.name}.csv'
        try:
            self.data = pd.read_csv(destination)
        except:
            self.data = load_dataset('copenlu/answerable_tydiqa')
            self.data = self.flatten(self.data)
            self.data = self.filter_language(self.data)
            self.explode_annotations(self.data)
            self.label_answerable(self.data)
            self.tokenize(self.data)
            self.clean(self.data)
            self.data = self.balance(self.data)
            self.data.to_csv(destination)
    
    def filter_language(self, dataset):
        return dataset[dataset["language"] == self.name] 
    
    def flatten(self, dataset):
        return dataset["train"].to_pandas().append(
            dataset["validation"].to_pandas()
        ) 
    
    def explode_annotations(self, dataset):
        dataset['answer_start'] = dataset['annotations'].apply(lambda x: x['answer_start'][0])
        dataset['answer_text'] = dataset['annotations'].apply(lambda x: x['answer_text'][0])  
        dataset.drop(columns=['annotations'], inplace=True)

    def tokenize(self, dataset):
        """Tokenize the dataset"""
        dataset['tokenized_question'] = dataset['question_text'].apply(self.tokenizer)
        dataset['tokenized_plaintext'] = dataset['document_plaintext'].apply(self.tokenizer)

    def label_answerable(self, dataset):
        """Label the dataset"""
        dataset['is_answerable'] = dataset['answer_start'].apply(lambda x: x != -1)

    def balance(self, dataset):
        """Balance the dataset"""
        positives = dataset[dataset['is_answerable']]
        negatives = dataset[~dataset['is_answerable']]
        smallest_class = min(len(positives), len(negatives))
        return positives.sample(n=smallest_class).append(negatives.sample(n=smallest_class))

    def clean(self, dataset):
        """Clean the dataset"""
        dataset['cleaned_question'] = dataset['tokenized_question'].apply(self.cleaner)
        dataset['cleaned_plaintext'] = dataset['tokenized_plaintext'].apply(self.cleaner)



    

