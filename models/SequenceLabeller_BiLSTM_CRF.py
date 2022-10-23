from .Model import Model
import numpy as np
import torch
from datasets import Dataset
from torch import nn
import io
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from torchcrf import CRF
from torch.optim.lr_scheduler import CyclicLR
from typing import Tuple
from tqdm.notebook import tqdm
from sklearn.metrics import precision_recall_fscore_support


class SequenceLabeller_BiLSTM_CRF(Model):

    def __init__(self, language, config):
        super().__init__(language, config)

        self.lstm_dim = 128
        self.dropout_prob = 0.1
        self.batch_size = 8
        self.lr = 1e-2
        self.n_epochs = 1
        self.n_workers = 0


    def _tag_token(self, token, index, answer_start, answer_end):
            """Tag a token with the IOB format"""
            if index == answer_start:
                return 2 # B
            elif answer_start < index <= answer_end:
                return 1 # I
            else:
                return 0 # O

    def _tag_sentence(self, sentence, answer_start, answer_end):
        """Tag a sentence with the IOB format"""
        return [
            self._tag_token(token, index, answer_start, answer_end)
            for index, token in enumerate(sentence)
        ]

    def _convert_to_iob(self, dataset):
        """Tag the dataset with the IOB format"""
        dataset['tags'] = dataset.apply(lambda row: self._tag_sentence(row['text'], row['answer_start'], row['answer_end']), axis=1)
        return dataset


    def extract_X(self, dataset, language: str = ""):
        """Extract features from the dataset"""
        dataset['text'] = dataset.apply(lambda row: np.concatenate((['<START>'], row['tokenized_question'], ['<SEP>'], row['tokenized_plaintext'], ['<STOP>'])), axis=1)
        dataset['answer_start'] = dataset.apply(lambda row: len(row['tokenized_question'])+2+row['answer_start'], axis=1)
        dataset['answer_end'] = dataset.apply(lambda row: len(row['tokenized_question'])+2+row['answer_end'], axis=1)
        dataset = self._convert_to_iob(dataset)

        train_dataset = Dataset.from_pandas(dataset[['text', 'tags']])
        vocabulary = set([t for s in train_dataset for t in s['text']])
        self.vocabulary, self.pretrained_embeddings = self._load_vectors('wiki-news-300d-1M.vec', vocabulary)
        self.tokenizer = FasttextTokenizer(self.vocabulary)


        # Create the model
        self.model = BiLSTM_CRF(
            pretrained_embeddings=torch.FloatTensor(self.pretrained_embeddings), 
            lstm_dim=self.lstm_dim, 
            dropout_prob=self.dropout_prob, 
            n_classes=3
        ).to(self.device)

        return train_dataset


    def extract_y(self, dataset, language: str = ""):
        """Extract the labels from the dataset"""
        return None

    # Reduce down to our vocabulary and word embeddings
    def _load_vectors(self, fname, vocabulary):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        tag_names = ['B', 'I', 'O']
        final_vocab = tag_names + ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
        final_vectors = [np.random.normal(size=(300,)) for _ in range(len(final_vocab))]
        for j,line in enumerate(fin):
            tokens = line.rstrip().split(' ')
            if tokens[0] in vocabulary or len(final_vocab) < 30000:
                final_vocab.append(tokens[0])
                final_vectors.append(np.array(list(map(float, tokens[1:]))))
        return final_vocab, np.vstack(final_vectors)


    def _collate_batch_bilstm(self, input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = [self.tokenizer.encode(i['text']) for i in input_data]
        seq_lens = [len(i) for i in input_ids]
        labels = [i['tags'] for i in input_data]

        max_length = max([len(i) for i in input_ids])

        input_ids = [(i + [0] * (max_length - len(i))) for i in input_ids]
        labels = [(i + [0] * (max_length - len(i))) for i in labels] # 0 is the id of the O tag

        assert (all(len(i) == max_length for i in input_ids))
        assert (all(len(i) == max_length for i in labels))
        return torch.tensor(input_ids), torch.tensor(seq_lens), torch.tensor(labels)


    def train(self, X, y):

        train_dl = DataLoader(X, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_batch_bilstm, num_workers=self.n_workers)
        # Create the optimizer
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = CyclicLR(optimizer, base_lr=0., max_lr=self.lr, step_size_up=1, step_size_down=len(train_dl)*self.n_epochs, cycle_momentum=False)

        # Keep track of the loss and best accuracy
        losses = []
        learning_rates = []

        # Iterate through epochs
        for ep in range(self.n_epochs):

            loss_epoch = []

            #Iterate through each batch in the dataloader
            for batch in tqdm(train_dl):
                # VERY IMPORTANT: Make sure the model is in training mode, which turns on 
                # things like dropout and layer normalization
                self.model.train()

                # VERY IMPORTANT: zero out all of the gradients on each iteration -- PyTorch
                # keeps track of these dynamically in its computation graph so you need to explicitly
                # zero them out
                optimizer.zero_grad()

                # Place each tensor on the GPU
                batch = tuple(t.to(self.device) for t in batch)
                input_ids = batch[0]
                seq_lens = batch[1]
                labels = batch[2]

                # Pass the inputs through the model, get the current loss and logits
                loss, logits = self.model(input_ids, seq_lens, labels=labels)
                losses.append(loss.item())
                loss_epoch.append(loss.item())

                # Calculate all of the gradients and weight updates for the model
                loss.backward()

                # Optional: clip gradients
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Finally, update the weights of the model
                optimizer.step()
                if scheduler != None:
                    scheduler.step()
                    learning_rates.append(scheduler.get_last_lr()[0])

        return losses, learning_rates

    def predict(self, X):

        valid_dl = DataLoader(X, batch_size=len(X), collate_fn=self._collate_batch_bilstm, num_workers=self.n_workers)

        tags_all = []

        # ALSO IMPORTANT: Don't accumulate gradients during this process
        with torch.no_grad():
            for batch in tqdm(valid_dl, desc='Evaluation'):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids = batch[0]
                seq_lens = batch[1]
                labels = batch[2]

                logits = self.model(input_ids, seq_lens, labels=labels)[-1]
                mask = (input_ids != 0)

                tags = self.model.decode(logits, mask)
                tags_all.extend([t for seq in tags for t in seq])

        return tags_all

    def evaluate(self, X, y):

        valid_dl = DataLoader(X, batch_size=len(X), collate_fn=self._collate_batch_bilstm, num_workers=self.n_workers)

        # VERY IMPORTANT: Put your model in "eval" mode -- this disables things like 
        # layer normalization and dropout
        self.model.eval()
        labels_all = []
        logits_all = []
        tags_all = []

        # ALSO IMPORTANT: Don't accumulate gradients during this process
        with torch.no_grad():
            for batch in tqdm(valid_dl, desc='Evaluation'):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids = batch[0]
                seq_lens = batch[1]
                labels = batch[2]

                logits = self.model(input_ids, seq_lens, labels=labels)[-1]
                mask = (input_ids != 0)
                labels_all.extend([l for seq,samp in zip(list(labels.detach().cpu().numpy()), input_ids) for l,i in zip(seq,samp) if i != 0])
                logits_all.extend(list(logits.detach().cpu().numpy()))

                tags = self.model.decode(logits, mask)
                tags_all.extend([t for seq in tags for t in seq])

        P, R, F1, _ = precision_recall_fscore_support(labels_all, tags_all, average='macro')
        return F1

    def save(self):
        torch.save(self.model.state_dict(), self.get_save_path('pt'))

    def load(self):
        self.model.load_state_dict(torch.load(self.get_save_path('pt')))



# Define the model
class BiLSTM_CRF(nn.Module):
    """
    Basic BiLSTM-CRF network
    """
    def __init__(
            self,
            pretrained_embeddings: torch.tensor,
            lstm_dim: int,
            dropout_prob: float = 0.1,
            n_classes: int = 2
    ):
        """
        Initializer for basic BiLSTM network
        :param pretrained_embeddings: A tensor containing the pretrained BPE embeddings
        :param lstm_dim: The dimensionality of the BiLSTM network
        :param dropout_prob: Dropout probability
        :param n_classes: The number of output classes
        """

        # First thing is to call the superclass initializer
        super(BiLSTM_CRF, self).__init__()

        # We'll define the network in a ModuleDict, which makes organizing the model a bit nicer
        # The components are an embedding layer, a 2 layer BiLSTM, and a feed-forward output layer
        self.model = nn.ModuleDict({
            'embeddings': nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=pretrained_embeddings.shape[0] - 1),
            'bilstm': nn.LSTM(
                pretrained_embeddings.shape[1],
                lstm_dim,
                2,
                batch_first=True,
                dropout=dropout_prob,
                bidirectional=True),
            'ff': nn.Linear(2*lstm_dim, n_classes),
            'CRF': CRF(n_classes, batch_first=True)
        })
        self.n_classes = n_classes

        # Initialize the weights of the model
        self._init_weights()

    def _init_weights(self):
        all_params = list(self.model['bilstm'].named_parameters()) + \
                     list(self.model['ff'].named_parameters())
        for n,p in all_params:
            if 'weight' in n:
                nn.init.xavier_normal_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)

    def forward(self, inputs, input_lens, labels = None):
        """
        Defines how tensors flow through the model
        :param inputs: (b x sl) The IDs into the vocabulary of the input samples
        :param input_lens: (b) The length of each input sequence
        :param labels: (b) The label of each sample
        :return: (loss, logits) if `labels` is not None, otherwise just (logits,)
        """

        # Get embeddings (b x sl x edim)
        embeds = self.model['embeddings'](inputs)

        # Pack padded: This is necessary for padded batches input to an RNN
        lstm_in = nn.utils.rnn.pack_padded_sequence(
            embeds,
            input_lens.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Pass the packed sequence through the BiLSTM
        lstm_out, hidden = self.model['bilstm'](lstm_in)

        # Unpack the packed sequence --> (b x sl x 2*lstm_dim)
        lstm_out,_ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Get emissions (b x seq_len x n_classes)
        emissions = self.model['ff'](lstm_out)
        outputs = (emissions,)
        if labels is not None:
            mask = (inputs != 0)
            # log-likelihood from the CRF
            log_likelihood = self.model['CRF'](emissions, labels, mask=mask, reduction='token_mean')
            outputs = (-log_likelihood,) + outputs

        return outputs

    def decode(self, emissions, mask):
        """
        Given a set of emissions and a mask, decode the sequence
        """
        return self.model['CRF'].decode(emissions, mask=mask)




class FasttextTokenizer:
    def __init__(self, vocabulary):
        self.vocab = {}
        for j,l in enumerate(vocabulary):
            self.vocab[l.strip()] = j

    def encode(self, text):
        # Text is assumed to be tokenized
        return [self.vocab[t] if t in self.vocab else self.vocab['[UNK]'] for t in text]