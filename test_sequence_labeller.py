from Preprocess import Preprocess
from Pipeline import Pipeline
from languages.English import English
from models.SequenceLabeller_BERT import SequenceLabeller
import numpy as np
from os import truncate
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from models.Model import Model
import torch
import evaluate

english = English()

preproccesor = Preprocess(english.tokenize, english.clean)
data = Pipeline().get_data(language=english.name, preproccesor=preproccesor)
train_data, validation_data = Pipeline().split_data(data)

sequence_labeller = SequenceLabeller('english')

X_train = sequence_labeller.extract_X(train_data, 'english')
X_val = sequence_labeller.extract_X(validation_data, 'english')

sequence_labeller.train(X_train, None)

sequence_labeller.save('english')

# sequence_labeller.load('english')

# print(validation_data)

# print(sequence_labeller.evaluate(X_val, None))

# print(sequence_labeller.predict(X_val))

# TODO: Implement/check that it can predict
# TODO: Implemement evaluation
# TODO: Implement beamsearch. look into num_beams