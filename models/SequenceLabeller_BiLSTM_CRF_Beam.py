from .Model import Model
import numpy as np
import torch
from datasets import Dataset
from torch import nn
import io
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from typing import Tuple
from tqdm.notebook import tqdm
from sklearn.metrics import precision_recall_fscore_support
import heapq


class SequenceLabeller_BiLSTM_CRF_Beam(Model):

    def __init__(self, language, config):
        super().__init__(language, config)

        self.lstm_dim = 128
        self.dropout_prob = 0.1
        self.batch_size = 8
        self.lr = 1e-2
        self.n_epochs = 1
        self.n_workers = 0
        self.beam_size = 2
   

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
        self.model = Seq2Seq(
            pretrained_embeddings=torch.FloatTensor(self.pretrained_embeddings), 
            lstm_dim=self.lstm_dim, 
            dropout_prob=self.dropout_prob, 
            n_classes=3,
            device=self.device,
            tokenizer=self.tokenizer,
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
                labels = batch[2]
                input_lens = batch[1]

                # Pass the inputs through the model, get the current loss and logits
                loss = self.model(input_ids, labels=labels, input_lens=input_lens)
                losses.append(loss.item())
                loss_epoch.append(loss.item())

                # Calculate all of the gradients and weight updates for the model
                loss.backward()

                # Optional: clip gradients
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Finally, update the weights of the model
                optimizer.step()

            return losses


    def _decode(self, inputs, input_lens, labels=None, beam_size=2):
        """
        Decoding/predicting the labels for an input text by running beam search.

        :param inputs: (b x sl) The IDs into the vocabulary of the input samples
        :param input_lens: (b) The length of each input sequence
        :param labels: (b) The label of each sample
        :param beam_size: the size of the beam 
        :return: predicted sequence of labels
        """

        assert inputs.shape[0] == 1

        softmax = nn.Softmax(dim=-1)

        # first, encode the input text
        encoder_output, encoder_hidden = self.model.model['encoder'](inputs, input_lens)
        decoder_hidden = encoder_hidden

        # the decoder starts generating after the Begining of Sentence (BOS) token
        decoder_input = torch.tensor([self.tokenizer.encode(['[BOS]',]),], device=self.device)
        target_length = labels.shape[1]
        
        # we will use heapq to keep top best sequences so far sorted in heap_queue 
        # these will be sorted by the first item in the tuple
        heap_queue = []
        heap_queue.append((torch.tensor(0), self.tokenizer.encode(['[BOS]']), decoder_input, decoder_hidden))

        # Beam Decoding
        for _ in range(target_length):
            # print("next len")
            new_items = []
            # for each item on the beam
            for j in range(len(heap_queue)): 
                # 1. remove from heap
                score, tokens, decoder_input, decoder_hidden = heapq.heappop(heap_queue)
                # 2. decode one more step
                decoder_output, decoder_hidden = self.model.model['decoder'](
                    decoder_input, decoder_hidden, torch.tensor([1]))
                decoder_output = softmax(decoder_output)
                # 3. get top-k predictions
                best_idx = torch.argsort(decoder_output[0], descending=True)[0]
                # print(decoder_output)
                # print(best_idx)
                for i in range(beam_size):
                    decoder_input = torch.tensor([[best_idx[i]]], device=self.device)
                    
                    new_items.append((score + decoder_output[0,0, best_idx[i]],
                                    tokens + [best_idx[i].item()], 
                                    decoder_input, 
                                    decoder_hidden))
            # add new sequences to the heap
            for item in new_items:
            # print(item)
                heapq.heappush(heap_queue, item)
            # remove sequences with lowest score (items are sorted in descending order)
            while len(heap_queue) > beam_size:
                heapq.heappop(heap_queue)
            
        final_sequence = heapq.nlargest(1, heap_queue)[0]
        assert labels.shape[1] == len(final_sequence[1][1:])
        return final_sequence



    def predict(self, X):

        valid_dl = DataLoader(X, batch_size=len(X), collate_fn=self._collate_batch_bilstm, num_workers=self.n_workers)

        tags_all = []

        with torch.no_grad():
            for batch in tqdm(valid_dl, desc='Evaluation'):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids = batch[0]
                input_lens = batch[1]
                labels = batch[2]

                best_seq = self._decode(input_ids, input_lens, labels=labels, beam_size=self.beam_size)

                tags_all += best_seq[1][1:]

        return tags_all

    def evaluate(self, X, y):

        valid_dl = DataLoader(X, batch_size=len(X), collate_fn=self._collate_batch_bilstm, num_workers=self.n_workers)

        # VERY IMPORTANT: Put your model in "eval" mode -- this disables things like 
        # layer normalization and dropout
        self.model.eval()
        labels_all = []
        tags_all = []

        # ALSO IMPORTANT: Don't accumulate gradients during this process
        with torch.no_grad():
            for batch in tqdm(valid_dl, desc='Evaluation'):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids = batch[0]
                input_lens = batch[1]
                labels = batch[2]

                best_seq = self._decode(input_ids, input_lens, labels=labels, beam_size=self.beam_size)
                mask = (input_ids != 0)
                labels_all.extend([l for seq,samp in zip(list(labels.detach().cpu().numpy()), input_ids) for l,i in zip(seq,samp) if i != 0])
                tags_all += best_seq[1][1:]
                # print(best_seq[1][1:], labels)
        P, R, F1, _ = precision_recall_fscore_support(labels_all, tags_all, average='macro')
        return F1

    def save(self, language: str):
        torch.save(self.model.state_dict(), f'{language}.pt')

    def load(self, language: str):
        self.model.load_state_dict(torch.load(f'{language}.pt'))



class EncoderRNN(nn.Module):
    """
    RNN Encoder model.
    """
    def __init__(self, 
            pretrained_embeddings: torch.tensor, 
            lstm_dim: int,
            dropout_prob: float = 0.1):
        """
        Initializer for EncoderRNN network
        :param pretrained_embeddings: A tensor containing the pretrained embeddings
        :param lstm_dim: The dimensionality of the LSTM network
        :param dropout_prob: Dropout probability
        """
        # First thing is to call the superclass initializer
        super(EncoderRNN, self).__init__()

        # We'll define the network in a ModuleDict, which makes organizing the model a bit nicer
        # The components are an embedding layer, and an LSTM layer.
        self.model = nn.ModuleDict({
            'embeddings': nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=pretrained_embeddings.shape[0] - 1),
            'lstm': nn.LSTM(pretrained_embeddings.shape[1], lstm_dim, 2, batch_first=True, bidirectional=True),
        })
        # Initialize the weights of the model
        self._init_weights()

    def _init_weights(self):
        all_params = list(self.model['lstm'].named_parameters())
        for n, p in all_params:
            if 'weight' in n:
                nn.init.xavier_normal_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)

    def forward(self, inputs, input_lens):
        """
        Defines how tensors flow through the model
        :param inputs: (b x sl) The IDs into the vocabulary of the input samples
        :param input_lens: (b) The length of each input sequence
        :return: (lstm output state, lstm hidden state) 
        """
        embeds = self.model['embeddings'](inputs)
        lstm_in = nn.utils.rnn.pack_padded_sequence(
                    embeds,
                    input_lens.cpu(),
                    batch_first=True,
                    enforce_sorted=False
                )
        lstm_out, hidden_states = self.model['lstm'](lstm_in)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        return lstm_out, hidden_states


class DecoderRNN(nn.Module):
    """
    RNN Decoder model.
    """
    def __init__(self, pretrained_embeddings: torch.tensor, 
            lstm_dim: int,
            dropout_prob: float = 0.1,
            n_classes: int = 2):
        """
        Initializer for DecoderRNN network
        :param pretrained_embeddings: A tensor containing the pretrained embeddings
        :param lstm_dim: The dimensionality of the LSTM network
        :param dropout_prob: Dropout probability
        :param n_classes: Number of prediction classes
        """
        # First thing is to call the superclass initializer
        super(DecoderRNN, self).__init__()
        # We'll define the network in a ModuleDict, which makes organizing the model a bit nicer
        # The components are an embedding layer, a LSTM layer, and a feed-forward output layer
        self.model = nn.ModuleDict({
            'embeddings': nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=pretrained_embeddings.shape[0] - 1),
            'lstm': nn.LSTM(pretrained_embeddings.shape[1], lstm_dim, 2, bidirectional=True, batch_first=True),
            'nn': nn.Linear(lstm_dim*2, n_classes),
        })
        # Initialize the weights of the model
        self._init_weights()      

    def forward(self, inputs, hidden, input_lens):
        """
        Defines how tensors flow through the model
        :param inputs: (b x sl) The IDs into the vocabulary of the input samples
        :param hidden: (b) The hidden state of the previous step
        :param input_lens: (b) The length of each input sequence
        :return: (output predictions, lstm hidden states) the hidden states will be used as input at the next step
        """
        embeds = self.model['embeddings'](inputs)

        lstm_in = nn.utils.rnn.pack_padded_sequence(
                    embeds,
                    input_lens.cpu(),
                    batch_first=True,
                    enforce_sorted=False
                )
        lstm_out, hidden_states = self.model['lstm'](lstm_in, hidden)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        output = self.model['nn'](lstm_out)
        return output, hidden_states

    def _init_weights(self):
        all_params = list(self.model['lstm'].named_parameters()) + list(self.model['nn'].named_parameters())
        for n, p in all_params:
            if 'weight' in n:
                nn.init.xavier_normal_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)


class FasttextTokenizer:
    def __init__(self, vocabulary):
        self.vocab = {}
        for j,l in enumerate(vocabulary):
            self.vocab[l.strip()] = j

    def encode(self, text):
        # Text is assumed to be tokenized
        return [self.vocab[t] if t in self.vocab else self.vocab['[UNK]'] for t in text]

# Define the model
class Seq2Seq(nn.Module):
    """
    Basic Seq2Seq network
    """
    def __init__(
            self,
            pretrained_embeddings: torch.tensor,
            lstm_dim: int,
            dropout_prob: float = 0.1,
            n_classes: int = 2,
            device: str = 'cpu',
            tokenizer: FasttextTokenizer = None
    ):
        """
        Initializer for basic Seq2Seq network
        :param pretrained_embeddings: A tensor containing the pretrained embeddings
        :param lstm_dim: The dimensionality of the LSTM network
        :param dropout_prob: Dropout probability
        :param n_classes: The number of output classes
        """

        # First thing is to call the superclass initializer
        super(Seq2Seq, self).__init__()

        # We'll define the network in a ModuleDict, which consists of an encoder and a decoder
        self.model = nn.ModuleDict({
            'encoder': EncoderRNN(pretrained_embeddings, lstm_dim, dropout_prob),
            'decoder': DecoderRNN(pretrained_embeddings, lstm_dim, dropout_prob, n_classes),
        })
        self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.5]+[1]*(3-1)).to(device)) # number of classes - 1

        self.tokenizer = tokenizer
        self.device = device


    def forward(self, inputs, input_lens, labels=None):
        """
        Defines how tensors flow through the model. 
        For the Seq2Seq model this includes 1) encoding the whole input text, 
        and running *target_length* decoding steps to predict the tag of each token.

        :param inputs: (b x sl) The IDs into the vocabulary of the input samples
        :param input_lens: (b) The length of each input sequence
        :param labels: (b) The label of each sample
        :return: (loss, logits) if `labels` is not None, otherwise just (logits,)
        """

        # Get embeddings (b x sl x embedding dim)
        encoder_output, encoder_hidden = self.model['encoder'](inputs, input_lens)
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([self.tokenizer.encode(['[BOS]'])]*inputs.shape[0], device=self.device)
        target_length = labels.size(1)

        loss = None
        for di in range(target_length):
            decoder_output, decoder_hidden = self.model['decoder'](
                decoder_input, decoder_hidden, torch.tensor([1]*inputs.shape[0]))

            if loss == None:   
                loss = self.loss(decoder_output.squeeze(1), labels[:, di])
            else:
                loss += self.loss(decoder_output.squeeze(1), labels[:, di])
            # Teacher forcing: Feed the target as the next input
            decoder_input = labels[:, di].unsqueeze(-1) 

        return loss / target_length