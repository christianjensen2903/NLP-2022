from os import pread
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from models.Model import Model
from datasets import Dataset
import numpy as np
import torch

# -- Resources --
# https://huggingface.co/Finnish-NLP/gpt2-finnish
# https://www.modeldifferently.com/en/2021/12/generación-de-fake-news-con-gpt-2/
# https://github.com/huggingface/transformers/issues/1528#issuecomment-544977912


class GPT2Generator(Model):
    def __init__(self):
        self.language_to_pretrained_name = {
            'english': 'gpt2',
            'finnish': 'Finnish-NLP/gpt2-finnish',
            'japanese': 'rinna/japanese-gpt2-small' # Unsure if japanese is functional (not tested)
        }

    def set_language(self, language):
        super().set_language(language)
        pretrained_name = self.language_to_pretrained_name[language]
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            pretrained_name
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(
            pretrained_name,
            output_hidden_states=True
        )

    def extract_X(self, dataset):
        train_dataset = Dataset.from_pandas(
            dataset[['question_text', 'document_plaintext']])

        def tokenize_function(examples):
            input_str = 'Question:' + \
                examples['question_text'] + 'Context:' + \
                examples['document_plaintext']
            # Truncating input_str to max length
            input_str = input_str[:1024]
            return self.tokenizer(
                input_str,
                padding=True
            )

        tokenized_train_dataset = train_dataset.map(
            tokenize_function,
            remove_columns=['question_text', 'document_plaintext'],
        )

        return tokenized_train_dataset

    def train(self, X):
        training_args = TrainingArguments(
            output_dir=self.get_save_path(),
            num_train_epochs=1,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=16,
            warmup_steps=200,
            weight_decay=0.01,
            prediction_loss_only=True,
            save_steps=10000
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=X
        )
        self.trainer.train()

    def generate_text(self, X, num_return_sequences=5, max_length=30):
        self.model.eval()
        text_ids = self.tokenizer.encode(X, return_tensors='pt')

        generated_text_samples = self.model.generate(
            text_ids,
            do_sample=True,
            max_length=max_length,
            num_return_sequences=num_return_sequences
        )

        for i, model_output in enumerate(generated_text_samples):
            print(
                f"{i}: {self.tokenizer.decode(model_output, skip_special_tokens=True)}"
            )

    def predict(self, X):
        self.model.eval()

        def get_last_hidden_state(row):
            row['last_hidden_state'] = self.model(
                torch.tensor(row['input_ids'])
            )[2][-1][-1]
            return row

        with torch.no_grad():
            return np.array(X.map(get_last_hidden_state)['last_hidden_state'])

    def save(self):
        path = self.get_save_path()
        self.trainer.save_model(path)
        self.tokenizer.save_pretrained(path)

    def load(self):
        path = self.get_save_path()
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        self.model = GPT2LMHeadModel.from_pretrained(path)
