from models.Model import Model
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SequenceLabeller2(Model):

    def __init__(self, language: str = ""):
        super().__init__()

        self.language = language

        self.start_tag = "<START>"
        self.sep_tag = "<SEP>"
        self.stop_tag = "<STOP>"
        self.embeeding_dim = 6
        self.hidden_dim = 6

        self.epoch = 300

        # self.tag_to_ix = {"B": 0, "I": 1, "O": 2, self.start_tag: 3, self.stop_tag: 4, self.sep_tag: 5}
        self.tag_to_ix = {"B": 0, "I": 1, "O": 2, self.start_tag: 3, self.stop_tag: 4}
        self.word_to_ix = {}

        self.model = BiLSTM_CRF(len(self.word_to_ix), self.tag_to_ix, self.embeeding_dim, self.hidden_dim, self.start_tag, self.stop_tag)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, weight_decay=1e-4)


    def _tag_token(self, token, index, answer_start, answer_end):
            """Tag a token with the IOB format"""
            if index == answer_start:
                return 'B'
            elif answer_start < index <= answer_end:
                return 'I'
            else:
                return 'O'

    def _tag_sentence(self, sentence, answer_start, answer_end):
        """Tag a sentence with the IOB format"""
        return [
            self._tag_token(token, index, answer_start, answer_end)
            for index, token in enumerate(sentence)
        ]

    def _convert_to_iob(self, dataset):
        """Tag the dataset with the IOB format"""
        return dataset.apply(lambda row: self._tag_sentence(row['tokenized_plaintext'], row['answer_start'], row['answer_end']), axis=1)

    
    def _prepare_sequence(self, seq):
        """Convert a sequence to a tensor of indices"""
        if self.word_to_ix == {}:
            raise ValueError('Word mapping not initialized')

        idxs = [self.word_to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    def _prepare_target(self, tags):
        """Convert a sequence of tags to a tensor of indices"""
        idxs = [self.tag_to_ix[t] for t in tags]
        return torch.tensor(idxs, dtype=torch.long)

    def _get_word_to_ix(self, dataset):
        """Map each word to a unique index"""
        for sentence in dataset:
            for word in sentence:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)

    def extract_X(self, dataset, language: str = ""):
        # Concatenate the question with the plaintext and tags
        # return dataset.apply(lambda x: np.concatenate(([self.start_tag],x['tokenized_question'], [self.sep_tag],x['tokenized_plaintext'], [self.stop_tag])), axis=1)
        return dataset['tokenized_plaintext']

    def extract_y(self, dataset, language: str = ""):
        tags = np.array(self._convert_to_iob(dataset))
        # y = tags.apply(lambda x: self._prepare_target(x))
        # y = [self._prepare_target(x) for x in tags]
        return tags


    def train(self, X, y):
        self._get_word_to_ix(X)

        for epoch in range(self.epoch):

            running_loss = 0.0
            n_nans = 0

            for sentence, tags in zip(X, y):
                # Reset the gradients
                self.model.zero_grad()

                # Turn our input into Tensors of word indices.
                sentence_in = self._prepare_sequence(sentence)
                targets = self._prepare_target(tags)

                # Run forward pass.
                loss = self.model.neg_log_likelihood(sentence_in, targets)

                # Compute the loss, gradients, and update the parameters
                loss.backward()
                self.optimizer.step()

                # Check for NaN
                if loss.item() == loss.item():
                    running_loss += loss.item()
                else:
                    n_nans += 1

            # Print statistics
            print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(X)))
            print('NaNs: %d' % n_nans)

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        pass

    def save(self, language: str):
        pass

    def load(self, language: str):
        pass



class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, start_tag, stop_tag):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.start_tag = start_tag
        self.stop_tag = stop_tag

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[self.start_tag], :] = -10000
        self.transitions.data[:, tag_to_ix[self.stop_tag]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[self.start_tag]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(self._log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.stop_tag]]
        alpha = self._log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.start_tag]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.stop_tag], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[self.start_tag]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self._argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.stop_tag]]
        best_tag_id = self._argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.start_tag]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        print(sentence)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


    def _argmax(self, vec):
        """return the argmax as a python int"""
        _, idx = torch.max(vec, 1)
        return idx.item()


    def _log_sum_exp(self,vec):
        """Compute log sum exp in a numerically stable way for the forward algorithm."""
        max_score = vec[0, self._argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
            torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))