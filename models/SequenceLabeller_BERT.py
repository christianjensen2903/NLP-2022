from models.Model import Model
import numpy as np
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
import wandb

class SequenceLabeller_BERT(Model):

    def __init__(self, language, config):
        super().__init__(language, config)

        if self.language == "english":
            self.model_name = "bert-base-uncased"
        elif self.language == "japanese":
            self.model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        elif self.language == "finnish":
            self.model_name = "TurkuNLP/bert-base-finnish-uncased-v1"
        elif self.language == "multilingual":
            self.model_name = "bert-base-multilingual-uncased"
        else:
            raise ValueError("Language not implemented")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=3).to(self.device)

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
        dataset['tags'] = dataset.apply(lambda row: self._tag_sentence(row['tokenized_plaintext'], row['answer_start'], row['answer_end']), axis=1)
        return dataset

    def _tokenize_and_align_labels(self, batch):
        """Realign the labels to the tokenized inputs. This is due to the tokenizer splitting the words into subwords"""
        tokenized_inputs = self.tokenizer(
                            batch['tokenized_question'],
                            batch['tokenized_plaintext'],
                            max_length=512,
                            padding='max_length',
                            truncation='only_second',
                            is_split_into_words=True,
                            return_tensors='pt' if self.device.type == 'cuda' else 'np'
                            )

        if self.device.type == 'cuda':
            tokenized_inputs = tokenized_inputs.to(self.device)
        
        labels = []
        for i, label in enumerate(batch["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            sep_index = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                elif sep_index == None: # Set all the tokens of the question to -100
                    label_ids.append(-100)
                    if tokenized_inputs['input_ids'][i][word_idx] == self.tokenizer.sep_token_id: # Find the index of the first SEP token to know where the question ends
                        sep_index = word_idx
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx-sep_index]) # The tags are offset by the number of tokens in the question
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        
        tokenized_inputs['labels'] = labels
        return tokenized_inputs


    def extract_X(self, dataset, language: str = ""):
        """Extract features from the dataset"""
        dataset = self._convert_to_iob(dataset)

        train_dataset = Dataset.from_pandas(dataset[['tokenized_question', 'tokenized_plaintext', 'tags']])
        train_dataset = train_dataset.map(self._tokenize_and_align_labels, batched=True)
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        return train_dataset


    def extract_y(self, dataset, language: str = ""):
        """Extract the labels from the dataset"""
        return None

    def train(self, X, y):
        """Train the model"""
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            report_to="wandb",              # Weights & Biases
            run_name='sequence-labeller',  # name of the W&B run (optional)
            num_train_epochs=6,              # total number of training epochs
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=16,
            warmup_steps=200,
            weight_decay=0.01,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=X, # X contains both the label and the features
            compute_metrics=self._compute_metrics
        )
        self.trainer.train()

        wandb.finish()


    def predict(self, X):
        """Predict the answer"""
        return self.trainer.predict(X)

    def _compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        binarizer = MultiLabelBinarizer().fit(true_labels)

        true_predictions = binarizer.transform(true_predictions)
        true_labels = binarizer.transform(true_labels)

        P, R, F1, _ = precision_recall_fscore_support(true_labels, true_predictions, average='macro', labels=[0,1,2])
        return {
            "precision": P,
            "recall": R,
            "f1": F1,
        }

    def evaluate(self, X, y):
        """Evaluate the model"""
        # predictions, labels, _ = self.predict(X)
        # predictions = np.argmax(predictions, axis=2)

        # # Remove ignored index (special tokens)
        # true_predictions = [
        #     [p for (p, l) in zip(prediction, label) if l != -100]
        #     for prediction, label in zip(predictions, labels)
        # ]
        # true_labels = [
        #     [l for (p, l) in zip(prediction, label) if l != -100]
        #     for prediction, label in zip(predictions, labels)
        # ]
        # binarizer = MultiLabelBinarizer().fit(true_labels)

        # true_predictions = binarizer.transform(true_predictions)
        # true_labels = binarizer.transform(true_labels)

        # P, R, F1, _ = precision_recall_fscore_support(true_labels, true_predictions, average='macro', labels=[0,1,2])
        # return {
        #     "precision": P,
        #     "recall": R,
        #     "f1": F1,
        # }
        return self.trainer.evaluate(X)

    def save(self):
        """Save the model"""
        self.trainer.save_model(self.get_save_path())

    def load(self):
        """Load the model"""
        self.model = AutoModelForTokenClassification.from_pretrained(self.get_save_path())
        self.trainer = Trainer(model=self.model)
