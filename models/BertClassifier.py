from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from models.Model import Model
import torch


class BertClassifier(Model):
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )

    def extract_X(self, dataset, language):
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True
        )

        encoded_dict = tokenizer.encode_plus(
            dataset['question_text'].tolist(),
            dataset['document_plaintext'].tolist(),
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return (encoded_dict['input_ids'], encoded_dict['attention_mask'])

    def train(self, X, y):
        print('hmm')
        input_ids, attention_masks = X
        dataset = TensorDataset(input_ids, attention_masks, y)
        batch_size = 32
        train_dataloader = DataLoader(
            dataset,
            sampler=RandomSampler(dataset),
            batch_size=batch_size
        )
        optimizer = AdamW(
            self.model.parameters(),
            lr=2e-5,
            eps=1e-8
        )
        epochs = 4
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        self.model.train()
        for epoch_i in range(0, epochs):
            print("")
            print(
                '======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            total_train_loss = 0
            for step, batch in enumerate(train_dataloader):

                if step % 40 == 0 and not step == 0:
                    print('Batch {:>5,}  of  {:>5,}.'.format(
                        step, len(train_dataloader)))

                self.model.zero_grad()

                b_input_ids = batch[0]
                b_input_mask = batch[1]
                b_labels = batch[2]
                loss, logits = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )

                total_train_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))

    def predict(self, X):
        """Predict the answer"""
        pass

    def evaluate(self, X, y):
        """Evaluate the model"""
        pass

    def save(self, language):
        self.model.save_pretrained(
            self.get_save_path(language, 'model')
        )

    def load(self, language):
        self.model.from_pretrained(
            self.get_save_path(language, 'model')
        )
