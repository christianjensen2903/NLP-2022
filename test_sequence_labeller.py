from Preprocess import Preprocess
from Pipeline import Pipeline
from languages.English import English
from models.SequenceLabeller import SequenceLabeller
import numpy as np
from os import truncate
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from models.Model import Model
import torch

english = English()

preproccesor = Preprocess(english.tokenize, english.clean)
data = Pipeline().get_data(language=english.name, preproccesor=preproccesor)

sequence_labeller = SequenceLabeller('english')

X = sequence_labeller.extract_X(data, 'english')
y = sequence_labeller.extract_y(data, 'english')

sequence_labeller.train(X, y)

sequence_labeller.save('english')