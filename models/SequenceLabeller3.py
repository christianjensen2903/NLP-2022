from models.Model import Model
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.Word2Vec import Word2Vec

class SequenceLabeller3(Model):

    def __init__(self, language: str = ""):
        super().__init__()

        self.language = language

        self.start_tag = "<START>"
        self.sep_tag = "<SEP>"
        self.stop_tag = "<STOP>"
        self.embedding_dim = -1 # will be set by word2vec model
        self.hidden_dim = 200 # hidden dimension of the LSTM
        self.char_mode = 'CNN' # 'CNN' or 'LSTM'
        self.use_gpu = torch.cuda.is_available()

        self.epoch = 50
        self.learning_rate = 0.015
        self.momentum = 0.9
        self.decay_rate = 0.05
        self.gradient_clip = 5.0

        # self.tag_to_ix = {"B": 0, "I": 1, "O": 2, self.start_tag: 3, self.stop_tag: 4, self.sep_tag: 5}
        # self.tag_to_ix = {"B": 0, "I": 1, "O": 2, self.start_tag: 3, self.stop_tag: 4}
        # self.word_to_ix = {}

        self.mappings_initialized = False


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

    def _create_dico(self, item_list):
        """
        Create a dictionary of items from a list of list of items.
        """
        # assert type(item_list) is list
        dico = {}
        for items in item_list:
            for item in items:
                if item not in dico:
                    dico[item] = 1
                else:
                    dico[item] += 1
        return dico

    def _create_mapping(self, dico):
        """
        Create a mapping (item to ID / ID to item) from a dictionary.
        Items are ordered by decreasing frequency.
        """
        sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
        id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
        item_to_id = {v: k for k, v in id_to_item.items()}
        return item_to_id, id_to_item

    def _word_mapping(self, words):
        """
        Create a dictionary and a mapping of words, sorted by frequency.
        """
        dico = self._create_dico(words)
        dico['<UNK>'] = 10000000 #UNK tag for unknown words
        word_to_id, id_to_word = self._create_mapping(dico)
        print("Found %i unique words (%i in total)" % (
            len(dico), sum(len(x) for x in words)
        ))
        return dico, word_to_id, id_to_word

    def _char_mapping(self, sentences):
        """
        Create a dictionary and mapping of characters, sorted by frequency.
        """
        chars = ["".join([w for w in s]) for s in sentences]
        dico = self._create_dico(chars)
        char_to_id, id_to_char = self._create_mapping(dico)
        print("Found %i unique characters" % len(dico))
        return dico, char_to_id, id_to_char

    def _tag_mapping(self, tags):
        """
        Create a dictionary and a mapping of tags, sorted by frequency.
        """
        dico = self._create_dico(tags)
        dico[self.start_tag] = -1
        dico[self.stop_tag] = -2
        tag_to_id, id_to_tag = self._create_mapping(dico)
        print("Found %i unique named entity tags" % len(dico))
        return dico, tag_to_id, id_to_tag

    def _prepare_dataset(self, dataset):
        """
        Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - character indexes
        - tag indexes
        """

        sentences = np.array(dataset.apply(lambda x: np.concatenate(([self.start_tag], x['tokenized_plaintext'], [self.stop_tag])), axis=1))
        # all_tags = np.array(dataset.apply(lambda x: np.concatenate(([self.start_tag], self._tag_sentence(x['tokenized_plaintext'], x['answer_start'], x['answer_end']), [self.stop_tag]))))
        all_tags = np.array(self._convert_to_iob(dataset).apply(lambda x: np.concatenate(([self.start_tag], x, [self.stop_tag]))))

        if self.mappings_initialized == False:
            self.dico_words, self.word_to_ix, self.ix_to_word = self._word_mapping(sentences)
            self.dico_chars, self.char_to_ix, self.ix_to_char = self._char_mapping(sentences)
            self.dico_tags, self.tag_to_ix, self.ix_to_tag = self._tag_mapping(all_tags)
            self.mappings_initialized = True

        data = []
        for sentence, tag in zip(sentences, all_tags):
            
            words = [self.word_to_ix[w] if w in self.word_to_ix else self.word_to_ix['<UNK>'] for w in sentence] #replace unknown words with UNK tag
            chars = [[self.char_to_ix[c] for c in w if c in self.char_to_ix] for w in sentence]
            tags = [self.tag_to_ix[t] for t in tag]

            data.append({
                'str_words': sentence,
                'words': words,
                'chars': chars,
                'tags': tags,
            })

        return data

    def _get_word2vec(self, dataset):
        """Retrieve word2vec model if it exists, otherwise create it"""
        word2vec = Word2Vec(self.language)
        try:
            word2vec.load(self.language)
        except:
            word2vec.train(dataset)
            word2vec.save(self.language)
        return word2vec

    def _get_word_embeddings(self, words, dataset):
        """Retrieve word embeddings"""
        word2vec = self._get_word2vec(dataset)
        self.embedding_dim = word2vec.model.vector_size # word2vec model dimension
        word2vec_embeddings = word2vec.predict(words)
        return word2vec_embeddings


    def extract_X(self, dataset, language: str = ""):
        # Concatenate the question with the plaintext and tags
        # return dataset.apply(lambda x: np.concatenate(([self.start_tag],x['tokenized_question'], [self.sep_tag],x['tokenized_plaintext'], [self.stop_tag])), axis=1)
        return self._prepare_dataset(dataset)

    def extract_y(self, dataset, language: str = ""):
        tags = np.array(self._convert_to_iob(dataset))
        # y = tags.apply(lambda x: self._prepare_target(x))
        # y = [self._prepare_target(x) for x in tags]
        return tags


    def train(self, X, y):
        word_embeddings = self._get_word_embeddings(self.word_to_ix, X)

        self.model = BiLSTM_CRF(vocab_size=len(self.word_to_ix),
                   tag_to_ix=self.tag_to_ix,
                   embedding_dim=self.embedding_dim,
                   hidden_dim=self.hidden_dim,
                   use_gpu=self.use_gpu,
                   char_to_ix=self.char_to_ix,
                   pre_word_embeds=word_embeddings,
                   use_crf=True,
                   char_mode=self.char_mode)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        #variables which will used in training process
        losses = [] #list to store all losses
        loss = 0.0 #Loss Initializatoin
        best_dev_F = -1.0 # Current best F-1 Score on Dev Set
        best_test_F = -1.0 # Current best F-1 Score on Test Set
        best_train_F = -1.0 # Current best F-1 Score on Train Set
        all_F = [[0, 0, 0]] # List storing all the F-1 Scores
        eval_every = len(X) # Calculate F-1 Score after this many iterations
        plot_every = 2000 # Store loss after this many iterations
        count = 0 #Counts the number of iterations

        for epoch in range(self.epoch):
            for i, index in enumerate(np.random.permutation(len(X))):
                count += 1
                data = X[index]

                ##gradient updates for each data entry
                self.model.zero_grad()

                sentence_in = data['words']
                sentence_in = Variable(torch.LongTensor(sentence_in))
                tags = data['tags']
                chars2 = data['chars']
                
                if self.char_mode == 'LSTM':
                    chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
                    d = {}
                    for i, ci in enumerate(chars2):
                        for j, cj in enumerate(chars2_sorted):
                            if ci == cj and not j in d and not i in d.values():
                                d[j] = i
                                continue
                    chars2_length = [len(c) for c in chars2_sorted]
                    char_maxl = max(chars2_length)
                    chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
                    for i, c in enumerate(chars2_sorted):
                        chars2_mask[i, :chars2_length[i]] = c
                    chars2_mask = Variable(torch.LongTensor(chars2_mask))
                
                if self.char_mode == 'CNN':

                    d = {}

                    ## Padding the each word to max word size of that sentence
                    chars2_length = [len(c) for c in chars2]
                    char_maxl = max(chars2_length)
                    chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
                    for i, c in enumerate(chars2):
                        chars2_mask[i, :chars2_length[i]] = c
                    chars2_mask = Variable(torch.LongTensor(chars2_mask))


                targets = torch.LongTensor(tags)

                #we calculate the negative log-likelihood for the predicted tags using the predefined function
                if self.use_gpu:
                    neg_log_likelihood = self.model.get_neg_log_likelihood(sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(), chars2_length, d)
                else:
                    neg_log_likelihood = self.model.get_neg_log_likelihood(sentence_in, targets, chars2_mask, chars2_length, d)
                loss += neg_log_likelihood.data / len(data['words'])
                neg_log_likelihood.backward()

                #we use gradient clipping to avoid exploding gradients
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

                #Storing loss
                if count % plot_every == 0:
                    loss /= plot_every
                    print(count, ': ', loss.item())
                    if losses == []:
                        losses.append(loss)
                    losses.append(loss)
                    loss = 0.0


                #Performing decay on the learning rate
                if count % len(X) == 0:
                    self._adjust_learning_rate(self.optimizer, lr=self.learning_rate/(1+self.decay_rate*count/len(X)))


    def _adjust_learning_rate(self, optimizer, lr):
        """
        shrink learning rate
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        pass

    def save(self, language: str):
        pass

    def load(self, language: str):
        pass



class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                 char_to_ix=None, pre_word_embeds=None, char_out_dimension=25,char_embedding_dim=25, use_gpu=False
                 , use_crf=True, char_mode='CNN', start_tag='<START>', stop_tag='<STOP>', dropout=0.5):
        '''
        Input parameters:
                
                vocab_size= Size of vocabulary (int)
                tag_to_ix = Dictionary that maps NER tags to indices
                embedding_dim = Dimension of word embeddings (int)
                hidden_dim = The hidden dimension of the LSTM layer (int)
                char_to_ix = Dictionary that maps characters to indices
                pre_word_embeds = Numpy array which provides mapping from word embeddings to word indices
                char_out_dimension = Output dimension from the CNN encoder for character
                char_embedding_dim = Dimension of the character embeddings
                use_gpu = defines availability of GPU, 
                    when True: CUDA function calls are made
                    else: Normal CPU function calls are made
                use_crf = parameter which decides if you want to use the CRF layer for output decoding
        '''
        
        super(BiLSTM_CRF, self).__init__()
        
        #parameter initialization for the model
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_out_dimension
        self.char_mode = char_mode
        self.start_tag = start_tag
        self.stop_tag = stop_tag

        if char_embedding_dim is not None:
            self.char_embedding_dim = char_embedding_dim
            
            #Initializing the character embedding layer
            self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
            self._init_embedding(self.char_embeds.weight)
            
            #Performing LSTM encoding on the character embeddings
            if self.char_mode == 'LSTM':
                self.char_lstm = nn.LSTM(char_embedding_dim, 25, num_layers=1, bidirectional=True)
                self._init_lstm(self.char_lstm)
                
            #Performing CNN encoding on the character embeddings
            if self.char_mode == 'CNN':
                
                layers = []
                
                layers.append(nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(3, char_embedding_dim), padding=(2,0)))
                
                layers.append(nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3, 1), padding_mode='replicate'))
                
                net = nn.Sequential(*layers)
                
                self.char_cnn3 = net

        #Creating Embedding layer with dimension of ( number of words * dimension of each word)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embeds is not None:
            #Initializes the word embeddings with pretrained word embeddings
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False
    
        #Initializing the dropout layer, with dropout specificed in parameters
        self.dropout = nn.Dropout(dropout)
        
        #Lstm Layer:
        #input dimension: word embedding dimension + character level representation
        #bidirectional=True, specifies that we are using the bidirectional LSTM
        if self.char_mode == 'LSTM':
            self.lstm = nn.LSTM(embedding_dim+25*2, hidden_dim, bidirectional=True)
        if self.char_mode == 'CNN':
            self.lstm = nn.LSTM(embedding_dim+self.out_channels, hidden_dim, bidirectional=True)
        
        #Initializing the lstm layer using predefined function for initialization
        self._init_lstm(self.lstm)
        
        # Linear layer which maps the output of the bidirectional LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)
        
        #Initializing the linear layer using predefined function for initialization
        self._init_linear(self.hidden2tag) 

        if self.use_crf:
            # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
            # Matrix has a dimension of (total number of tags * total number of tags)
            self.transitions = nn.Parameter(
                torch.zeros(self.tagset_size, self.tagset_size))
            
            # These two statements enforce the constraint that we never transfer
            # to the start tag and we never transfer from the stop tag
            self.transitions.data[tag_to_ix[self.start_tag], :] = -10000
            self.transitions.data[:, tag_to_ix[self.stop_tag]] = -10000

    def _init_embedding(self, input_embedding):
        """
        Initialize embedding
        """
        bias = np.sqrt(3.0 / input_embedding.size(1))
        nn.init.uniform(input_embedding, -bias, bias)

    def _init_linear(self, input_linear):
        """
        Initialize linear transformation
        """
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    def _init_lstm(self, input_lstm):
        """
        Initialize lstm
        
        PyTorch weights parameters:
        
            weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
                of shape `(hidden_size * input_size)` for `k = 0`. Otherwise, the shape is
                `(hidden_size * hidden_size)`
                
            weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
                of shape `(hidden_size * hidden_size)`            
        """
        
        # Weights init for forward layer
        for ind in range(0, input_lstm.num_layers):
            
            ## Gets the weights Tensor from our model, for the input-hidden weights in our current layer
            weight = eval('input_lstm.weight_ih_l' + str(ind))
            
            # Initialize the sampling range
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            
            # Randomly sample from our samping range using uniform distribution and apply it to our current layer
            nn.init.uniform(weight, -sampling_range, sampling_range)
            
            # Similar to above but for the hidden-hidden weights of the current layer
            weight = eval('input_lstm.weight_hh_l' + str(ind))
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -sampling_range, sampling_range)
            
            
        # We do the above again, for the backward layer if we are using a bi-directional LSTM (our final model uses this)
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
                sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
                nn.init.uniform(weight, -sampling_range, sampling_range)
                weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
                sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
                nn.init.uniform(weight, -sampling_range, sampling_range)

        # Bias initialization steps
        
        # We initialize them to zero except for the forget gate bias, which is initialized to 1
        if input_lstm.bias:
            for ind in range(0, input_lstm.num_layers):
                bias = eval('input_lstm.bias_ih_l' + str(ind))
                
                # Initializing to zero
                bias.data.zero_()
                
                # This is the range of indices for our forget gates for each LSTM cell
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                
                #Similar for the hidden-hidden layer
                bias = eval('input_lstm.bias_hh_l' + str(ind))
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                
            # Similar to above, we do for backward layer if we are using a bi-directional LSTM 
            if input_lstm.bidirectional:
                for ind in range(0, input_lstm.num_layers):
                    bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                    bias.data.zero_()
                    bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                    bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                    bias.data.zero_()
                    bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

    def _log_sum_exp(self, vec):
        '''
        This function calculates the score for the forward algorithm
        vec 2D: 1 * tagset_size
        '''
        max_score = vec[0, self._argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
        
    def _argmax(self, vec):
        '''
        This function returns the max index in a vector
        '''
        _, idx = torch.max(vec, 1)
        return self._to_scalar(idx)

    def _to_scalar(self, var):
        '''
        Function to convert pytorch tensor to a scalar
        '''
        return var.view(-1).data.tolist()[0]

    def _score_sentence(self, feats, tags):
        '''
        Gives the score of a provided tag sequence
        tags is ground_truth, a list of ints, length is len(sentence)
        feats is a 2D tensor, len(sentence) * tagset_size
        '''
        r = torch.LongTensor(range(feats.size()[0]))
        if self.use_gpu:
            r = r.cuda()
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_to_ix[self.start_tag]]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_to_ix[self.stop_tag]])])
        else:
            pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_ix[self.start_tag]]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_ix[self.stop_tag]])])

        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

        return score

    def _forward_alg(self, feats):
        '''
        This function performs the forward algorithm
        '''
        # calculate in log domain
        # feats is len(sentence) * tagset_size
        # initialize alpha with a Tensor with values all equal to -10000.
        
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[self.start_tag]] = 0.
        
        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)
        if self.use_gpu:
            forward_var = forward_var.cuda()
            
        # Iterate through the sentence
        for feat in feats:
            # broadcast the emission score: it is the same regardless of
            # the previous tag
            emit_score = feat.view(-1, 1)
            
            # the ith entry of trans_score is the score of transitioning to
            # next_tag from i
            tag_var = forward_var + self.transitions + emit_score
            
            # The ith entry of next_tag_var is the value for the
            # edge (i -> next_tag) before we do log-sum-exp
            max_tag_var, _ = torch.max(tag_var, dim=1)
            
            # The forward variable for this tag is log-sum-exp of all the
            # scores.
            tag_var = tag_var - max_tag_var.view(-1, 1)
            
            # Compute log sum exp in a numerically stable way for the forward algorithm
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1) # ).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag_to_ix[self.stop_tag]]).view(1, -1)
        alpha = self._log_sum_exp(terminal_var)
        # Z(x)
        return alpha


    def viterbi_algo(self, feats):
        '''
        In this function, we implement the viterbi algorithm explained above.
        A Dynamic programming based approach to find the best tag sequence
        '''
        backpointers = []
        # analogous to forward
        
        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[self.start_tag]] = 0
        
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = Variable(init_vvars)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy() # holds the backpointers for this step
            next_tag_var = next_tag_var.data.cpu().numpy() 
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t] # holds the viterbi variables for this step
            viterbivars_t = Variable(torch.FloatTensor(viterbivars_t))
            if self.use_gpu:
                viterbivars_t = viterbivars_t.cuda()
                
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.stop_tag]]
        terminal_var.data[self.tag_to_ix[self.stop_tag]] = -10000.
        terminal_var.data[self.tag_to_ix[self.start_tag]] = -10000.
        best_tag_id = self._argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]
        
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
            
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.start_tag] # Sanity check
        best_path.reverse()
        return path_score, best_path


    def forward(self, sentence, chars, chars2_length, d): 
        '''
        The function calls viterbi decode and generates the 
        most probable sequence of tags for the sentence
        '''
        
        # Get the emission scores from the BiLSTM
        feats = self._get_lstm_features(sentence, chars, chars2_length, d)
        # viterbi to get tag_seq
        
        # Find the best path, given the features.
        if self.use_crf:
            score, tag_seq = self.viterbi_decode(feats)
        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = list(tag_seq.cpu().data)

        return score, tag_seq

    def _get_lstm_features(self, sentence, chars2, chars2_length, d):
        '''
        The get_lstm_features function returns the LSTM's tag vectors. The function performs all the steps mentioned above for the model.
        Steps:

        1. It takes in characters, converts them to embeddings using our character CNN.
        2. We concat Character Embeeding with glove vectors, use this as features that we feed to Bidirectional-LSTM.
        3. The Bidirectional-LSTM generates outputs based on these set of features.
        4. The output are passed through a linear layer to convert to tag space.
        '''
        if self.char_mode == 'LSTM':
            
                chars_embeds = self.char_embeds(chars2).transpose(0, 1)
                
                packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)
                
                lstm_out, _ = self.char_lstm(packed)
                
                outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
                
                outputs = outputs.transpose(0, 1)
                
                chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))
                
                if self.use_gpu:
                    chars_embeds_temp = chars_embeds_temp.cuda()
                
                for i, index in enumerate(output_lengths):
                    chars_embeds_temp[i] = torch.cat((outputs[i, index-1, 25], outputs[i, 0, 25:]))
                
                chars_embeds = chars_embeds_temp.clone()
                
                for i in range(chars_embeds.size(0)):
                    chars_embeds[d[i]] = chars_embeds_temp[i]
        
        
        if self.char_mode == 'CNN':
            chars_embeds = self.char_embeds(chars2).unsqueeze(1)

            ## Creating Character level representation using Convolutional Neural Netowrk
            ## followed by a Maxpooling Layer
            chars_cnn_out3 = self.char_cnn3(chars_embeds)
            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                                                kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)

            ## Loading word embeddings
        embeds = self.word_embeds(sentence)

        ## We concatenate the word embeddings and the character level representation
        ## to create unified representation for each word
        embeds = torch.cat((embeds, chars_embeds), 1)

        embeds = embeds.unsqueeze(1)

        ## Dropout on the unified embeddings
        embeds = self.dropout(embeds)

        ## Word lstm
        ## Takes words as input and generates a output at each step
        lstm_out, _ = self.lstm(embeds)

        ## Reshaping the outputs from the lstm layer
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim*2)

        ## Dropout on the lstm output
        lstm_out = self.dropout(lstm_out)

        ## Linear layer converts the ouput vectors to tag space
        lstm_feats = self.hidden2tag(lstm_out)
        
        return lstm_feats

    def get_neg_log_likelihood(self, sentence, tags, chars2, chars2_length, d):
        '''
        The function returns the negative log likelihood of the sentence.
        '''
        # sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.tagset_size
        feats = self._get_lstm_features(sentence, chars2, chars2_length, d)

        if self.use_crf:
            forward_score = self._forward_alg(feats)
            gold_score = self._score_sentence(feats, tags)
            return forward_score - gold_score
        else:
            tags = Variable(tags)
            scores = nn.functional.cross_entropy(feats, tags)
            return scores