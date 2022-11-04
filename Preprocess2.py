import pandas as pd


class Preprocess2:
    def __init__(self, tokenizer, cleaner):
        self.tokenizer = tokenizer
        self.cleaner = cleaner
        self.from_datasets = True

    def preprocess(self, dataset):
        """Preprocess the dataset"""
        data = dataset
        if self.from_datasets:
            data = self.flatten(dataset)
            data = self.explode_annotations(data)
            data = data.dropna()
            data = data.drop_duplicates()
            data = self.label_answerable(data)
        data = self.tokenize(data)
        data = self.clean(data)
        if self.from_datasets:
            data = self.balance(data)

        return data

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