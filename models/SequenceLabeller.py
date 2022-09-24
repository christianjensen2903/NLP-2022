from models.Model import Model
import numpy as np
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from torch.utils.data import TensorDataset
import torch
from datasets import Dataset

class SequenceLabeller(Model):

    def __init__(self, language: str = ""):
        super().__init__()
        if language == "english":
            self.model_name = "distilbert-base-uncased"
        elif language == "japanese":
            self.model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        elif language == "finnish":
            self.model_name = "TurkuNLP/bert-base-finnish-cased-v1"
        else:
            raise ValueError("Language not implemented")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=3)

    def _tag_token(self, token, index, answer_start, answer_end):
            """Tag a token with the IOB format"""
            if index == answer_start:
                return 0 # B
            elif answer_start < index <= answer_end:
                return 1 # I
            else:
                return 2 # O

    def _tag_sentence(self, sentence, answer_start, answer_end):
        """Tag a sentence with the IOB format"""
        return [
            self._tag_token(token, index, answer_start, answer_end)
            for index, token in enumerate(sentence)
        ]

    def _convert_to_iob(self, dataset):
        """Tag the dataset with the IOB format"""

        dataset['plaintext_tags'] = dataset.apply(lambda row: self._tag_sentence(row['tokenized_plaintext'], row['answer_start'], row['answer_end']), axis=1)

        return dataset

    def _tokenize(self, examples):
        """Convert the dataset to a format that the model can understand using the tokenizer"""
        tokenized_inputs = self.tokenizer(examples['tokenized_question'], examples['tokenized_plaintext'], is_split_into_words=True, truncation='only_second') # padding='max_length', max_length=512, return_tensors='pt')
        labels = self._realign_labels(examples['plaintext_tags'], tokenized_inputs)
        tokenized_inputs['labels'] = labels
        return tokenized_inputs


    def extract_X(self, dataset, language: str = ""):
        """Extract features from the dataset"""
        dataset = self._convert_to_iob(dataset)

        train_dataset = Dataset.from_pandas(dataset[['tokenized_question', 'tokenized_plaintext', 'plaintext_tags']])
        # X = dataset.apply(lambda row: np.concatenate([['[CLS]'], row['tokenized_question'], ['[SEP]'], row['tokenized_plaintext']]), axis=1)
        # tokenized_inputs = self.tokenizer(dataset['tokenized_question'].tolist(), dataset['tokenized_plaintext'].tolist(), is_split_into_words=True, truncation='only_second') # padding='max_length', max_length=512, return_tensors='pt')
        train_dataset = train_dataset.map(self._tokenize, remove_columns=['tokenized_question', 'tokenized_plaintext', 'plaintext_tags'])
        return train_dataset

    def _realign_labels(self, tags, tokenized_inputs):
        """Realign the labels to the tokenized inputs. This is due to the tokenizer splitting the words into subwords"""

        sep_index = None

        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif sep_index == None: # Set all the tokens of the question to -100
                label_ids.append(-100)
                if tokenized_inputs['input_ids'][word_idx] == self.tokenizer.sep_token_id: # Find the index of the first SEP token to know where the question ends
                    sep_index = word_idx
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(tags[word_idx-sep_index]) # The tags are offset by the number of tokens in the question
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        return label_ids


    def extract_y(self, dataset, language: str = ""):
        """Extract the labels from the dataset"""
        y = dataset.apply(lambda row: np.concatenate([[-100]*(2+len(row['tokenized_question'])), row['plaintext_tags'], [-100]]), axis=1).to_numpy() # -100 is the padding token and the plus 2 is for the CLS and SEP tags
        return y

    def train(self, X, y):
        """Train the model"""
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            #output_dir='./',          # output directory
            num_train_epochs=1,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=16,
            warmup_steps=200,
            weight_decay=0.01,
            prediction_loss_only=True,
            save_steps=10000
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=X # X contains both the label and the features
        )
        self.trainer.train()


    def predict(self, X):
        """Predict the answer"""
        return self.trainer.predict(X)

    def evaluate(self, X, y):
        """Evaluate the model"""
        return self.trainer.evaluate(X, y)

    def save(self, language: str):
        """Save the model"""
        self.trainer.save_model(f'./saved_models/SequenceLabeller/{language}')
        # self.trainer.save_model(self.get_save_path(language, "pth"))

    def load(self, language: str):
        """Load the model"""
        self.model = AutoModelForTokenClassification.from_pretrained(f'./saved_models/SequenceLabeller/{language}')
        self.trainer = Trainer(model=self.model)
