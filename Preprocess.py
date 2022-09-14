import pandas as pd


class Preprocess:
    def __init__(self, data, tokenizer, cleaner):
        self.data = data
        self.tokenizer = tokenizer
        self.cleaner = cleaner

    def preprocess(self):
        """Preprocess the dataset"""
        self.data = self.flatten(self.data)
        self.data = self.explode_annotations(self.data)
        self.data = self.data.dropna()
        self.data = self.data.drop_duplicates()
        self.data = self.label_answerable(self.data)
        self.data = self.tokenize(self.data)
        self.data = self.clean(self.data)
        self.data = self.data = self.balance(self.data)

        return self.data

    def flatten(self, dataset):
        """Concat the training data and the validation data"""
        return dataset["train"].to_pandas().append(
            dataset["validation"].to_pandas()
        ) 
    
    def explode_annotations(self, dataset):
        """Explode the annotations from json to columns"""
        dataset['answer_start'] = dataset['annotations'].apply(lambda x: x['answer_start'][0])
        dataset['answer_text'] = dataset['annotations'].apply(lambda x: x['answer_text'][0])  
        dataset.drop(columns=['annotations'], inplace=True)
        return dataset

    def tokenize(self, dataset):
        """Tokenize the dataset"""
        dataset['tokenized_question'] = dataset['question_text'].apply(self.tokenizer)
        dataset['tokenized_plaintext'] = dataset['document_plaintext'].apply(self.tokenizer)
        return dataset

    def label_answerable(self, dataset):
        """Label the dataset"""
        dataset['is_answerable'] = dataset['answer_start'].apply(lambda x: x != -1)
        return dataset

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
        return dataset