from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer
from models.Model import Model
import torch


class BertClassifier(Model):
    def extract_X(self, dataset, language):
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True
        )
        input_ids = []
        attention_masks = []
        sentences = dataset['tokenized_question']
        context = dataset['tokenized_question']

        # For every sentence...
        for i in range(len(sentences)):
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                sentences[i],                      # Sentence to encode.
                context[i],
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=64,           # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,   # Construct attn. masks.
                return_tensors='pt',     # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return (input_ids, attention_masks)

    def train(self, X, y):
        """Train the model"""
        pass

    def predict(self, X):
        """Predict the answer"""
        pass

    def evaluate(self, X, y):
        """Evaluate the model"""
        pass

    def save(self, language):
        """Save the model"""
        pass

    def load(self, language):
        """Load the model"""
        pass
    
    def explainability(self):
        """Use an use an interpretability method on the model"""
        pass


# Load the BERT tokenizer.

# Tokenize all of the sentences and map the tokens to thier word IDs.
labels = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_masks, labels)
# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=2,  # The number of output lafbels--2 for binary classification.
    # You can increase this for multi-class tasks.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()
