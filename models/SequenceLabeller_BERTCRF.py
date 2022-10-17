from models.Model import Model
import numpy as np
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from torch.utils.data import TensorDataset
import torch
from datasets import Dataset
import evaluate
import wandb
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import  TokenClassifierOutput
from torch import nn
from torch.nn import CrossEntropyLoss
import torch
import torchcrf
from typing import List, Optional

# https://github.com/shushanxingzhe/transformers_ner

class SequenceLabeller_BERTCRF(Model):

    def __init__(self, language: str = "", num_beams=-1):
        super().__init__()
        if language == "english":
            self.model_name = "bert-base-uncased"
        elif language == "japanese":
            self.model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        elif language == "finnish":
            self.model_name = "TurkuNLP/bert-base-finnish-cased-v1"
        else:
            raise ValueError("Language not implemented")

        self.device = "mps" if torch.has_mps else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=3)
        self.model = BertCRF.from_pretrained(self.model_name, num_labels=3, num_beams=num_beams).to(self.device)

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

        # dataset['plaintext_tags'] = dataset.apply(lambda row: self._tag_sentence(row['tokenized_plaintext'], row['answer_start'], row['answer_end']), axis=1)
        dataset['tags'] = dataset.apply(lambda row: self._tag_sentence(row['text'], row['answer_start'], row['answer_end']), axis=1)
        return dataset

    def _tokenize(self, examples):
        """Convert the dataset to a format that the model can understand using the tokenizer"""
        tokenized_inputs = self.tokenizer(examples['text'], is_split_into_words=True, truncation=True, padding='max_length', max_length=512, add_special_tokens=False) # , return_tensors='pt'
        labels = self._realign_labels(examples['tags'], tokenized_inputs)
        tokenized_inputs['labels'] = labels
        return tokenized_inputs


    def extract_X(self, dataset, language: str = ""):
        """Extract features from the dataset"""
        dataset['text'] = dataset.apply(lambda row: np.concatenate((['<START>'], row['tokenized_question'], ['<SEP>'], row['tokenized_plaintext'], ['<STOP>'])), axis=1)
        dataset['answer_start'] = dataset.apply(lambda row: len(row['tokenized_question'])+2+row['answer_start'], axis=1)
        dataset['answer_end'] = dataset.apply(lambda row: len(row['tokenized_question'])+2+row['answer_end'], axis=1)
        dataset = self._convert_to_iob(dataset)

        train_dataset = Dataset.from_pandas(dataset[['text', 'tags']])
        # X = dataset.apply(lambda row: np.concatenate([['[CLS]'], row['tokenized_question'], ['[SEP]'], row['tokenized_plaintext']]), axis=1)
        # tokenized_inputs = self.tokenizer(dataset['tokenized_question'].tolist(), dataset['tokenized_plaintext'].tolist(), is_split_into_words=True, truncation='only_second') # padding='max_length', max_length=512, return_tensors='pt')
        train_dataset = train_dataset.map(self._tokenize, remove_columns=['text', 'tags'])
        return train_dataset

    def _realign_labels(self, tags, tokenized_inputs):
        """Realign the labels to the tokenized inputs. This is due to the tokenizer splitting the words into subwords"""

        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to O
            if word_idx is None:
                label_ids.append(2)
            # elif sep_index == None: # Set all the tokens of the question to -100
            #     label_ids.append(-100)
            #     if tokenized_inputs['input_ids'][word_idx] == self.tokenizer.sep_token_id: # Find the index of the first SEP token to know where the question ends
            #         sep_index = word_idx
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(tags[word_idx]) # The tags are offset by the number of tokens in the question
            else:
                label_ids.append(tags[previous_word_idx])
                # label_ids.append(-100)
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
            output_dir='./results',          # output directory
            report_to="wandb",              # Weights & Biases
            run_name='sequence-labeller',  # name of the W&B run (optional)
            num_train_epochs=20,              # total number of training epochs
            learning_rate=2e-5,
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
            train_dataset=X, # X contains both the label and the features
            compute_metrics=self._compute_metrics
        )
        self.trainer.train()

        wandb.finish()


    def predict(self, X):
        """Predict the answer"""
        return self.trainer.predict(X)

    def _compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        label_list = ['B', 'I', 'O']

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        metric = evaluate.load("seqeval")

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def evaluate(self, X, y):
        """Evaluate the model"""
        predictions, labels, _ = self.predict(X)
        predictions = np.argmax(predictions, axis=2)

        label_list = ['B', 'I', 'O']

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        metric = evaluate.load("seqeval")

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return results
        # return self.trainer.evaluate(X)

    def save(self, language: str):
        """Save the model"""
        self.trainer.save_model(f'./saved_models/SequenceLabeller/{language}')
        # self.trainer.save_model(self.get_save_path(language, "pth"))

    def load(self, language: str):
        """Load the model"""
        self.model = AutoModelForTokenClassification.from_pretrained(f'./saved_models/SequenceLabeller/{language}')
        self.trainer = Trainer(model=self.model)


class BertCRF(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_beams = config.num_beams

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            log_likelihood, tags = self.crf(logits, labels), self.crf.decode(logits, num_beams=self.num_beams)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits, num_beams=self.num_beams)
        tags = torch.Tensor(tags)

        if not return_dict:
            output = (tags,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, tags


class CRF(torchcrf.CRF):

    def __init__(self, num_tags: int, batch_first: bool = False):
        super().__init__(num_tags, batch_first)

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None, num_beams = -1) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if num_beams == -1:
            return self._viterbi_decode(emissions, mask)
        else:
            return self._beam_search(emissions, mask, num_beams)

    def _beam_search(self, emissions: torch.Tensor, mask: torch.ByteTensor, num_beams: int) -> List[List[int]]:
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()
        assert num_beams > 0 and num_beams <= self.num_tags

        seq_length, batch_size = mask.shape

        for l in range(0, batch_size):
            # Get the emission scores for each sequence in the batch
            # shape: (seq_length, num_tags)
            sequence_emission = emissions[:, l, :]

            # Start transition and first emission
            # shape: (num_tags, 1)
            start_scores = self.start_transitions + sequence_emission[0]

            # Prune the start scores to only the top num_beams
            # shape: (batch_size, num_beams)
            top_k_scores, top_k_tags = start_scores.topk(num_beams)
            beam = [([top_k_tags[i].item()], top_k_scores[i]) for i in range(num_beams)]
            history = [beam]

            # score is a tensor of size (batch_size, num_tags) where for every batch,
            # value at column j stores the score of the best tag sequence so far that ends
            # with tag j
            # history saves where the best tags candidate transitioned from; this is used
            # when we trace back the best tag sequence

            # Beam search algorithm recursive case: we compute the score of the best k tag sequences
            # for every possible next tag
            for i in range(1, seq_length):

                candidates = []

                for (prev, prev_score) in beam:

                    # Broadcast the previous score to all tags
                    # shape: (num_tags)
                    broadcast_prev_score = prev_score.expand(self.num_tags)

                    # Broadcast emission scores for the current timestep
                    # shape: (num_tags)
                    broadcast_emission = sequence_emission[i]

                    # Compute the score of transitioning from previous tag to all tags
                    # shape: (num_tags)
                    # print(f'Prev: {prev}')
                    # print(f'Transition: {self.transitions[prev]}')
                    # print(f'Broadcast prev score: {broadcast_prev_score}')
                    # print(f'Broadcast emission: {broadcast_emission}')
                    next_score = self.transitions[prev][0] + broadcast_emission + broadcast_prev_score
                    # print(f'Next score: {next_score}')

                    # Add to candidates
                    for j in range(self.num_tags):
                        candidates.append((prev + [j], next_score[j])) 

                # Keep the top k candidates
                # shape: (num_beams)
                top_k_scores, top_k_ind = torch.stack([c[1] for c in candidates]).topk(num_beams)

                beam = [(candidates[top_k_ind[j].item()][0], top_k_scores[j]) for j in range(num_beams)]

                history.append(beam)

        # Return sequence with max score
        # shape: (batch_size, seq_length)
        best_beam = sorted(beam, key=lambda x: -x[1])[0][0]
        return best_beam