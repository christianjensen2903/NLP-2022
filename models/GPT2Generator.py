# from os import truncate
# from transformers import BertForSequenceClassification, AdamW
# from torch.utils.data import DataLoader, RandomSampler
# from torch.utils.data import TensorDataset
# from transformers import BertTokenizer, get_linear_schedule_with_warmup
# from models.Model import Model
# import torch


# class BertClassifier(Model):
#     def __init__(self):
#         self.model = BertForSequenceClassification.from_pretrained(
#             "bert-base-uncased",
#             num_labels=2,
#             output_attentions=False,
#             output_hidden_states=False,
#         )

#     def extract_X(self, dataset, language):
#         tokenizer = BertTokenizer.from_pretrained(
#             'bert-base-uncased',
#             do_lower_case=True
#         )
#         input_ids = []
#         attention_masks = []

#         for index, row in dataset.iterrows():
#             encoded_dict = tokenizer.encode_plus(
#                 row['question_text'],
#                 row['document_plaintext'],
#                 add_special_tokens=True,
#                 max_length=512,
#                 padding='max_length',
#                 truncation=True,
#                 return_attention_mask=True,
#                 return_tensors='pt',
#             )
#             input_ids.append(encoded_dict['input_ids'])
#             attention_masks.append(encoded_dict['attention_mask'])

#         input_ids = torch.cat(input_ids, dim=0)
#         attention_masks = torch.cat(attention_masks, dim=0)
#         return (input_ids, attention_masks)

#     def train(self, X, y):
#         input_ids, attention_masks = X
#         y = torch.tensor(y.values, dtype=torch.long)
#         formatted_y = torch.zeros((len(y), 2))
#         formatted_y[torch.arange(len(y)), y] = 1
#         dataset = TensorDataset(input_ids, attention_masks, formatted_y)
#         batch_size = 32
#         train_dataloader = DataLoader(
#             dataset,
#             sampler=RandomSampler(dataset),
#             batch_size=batch_size
#         )
#         optimizer = AdamW(
#             self.model.parameters(),
#             lr=2e-5,
#             eps=1e-8
#         )
#         epochs = 4
#         total_steps = len(train_dataloader) * epochs
#         scheduler = get_linear_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=0,
#             num_training_steps=total_steps
#         )

#         self.model.train()
#         for epoch_i in range(0, epochs):
#             print("")
#             print(
#                 '======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
#             print('Training...')

#             total_train_loss = 0
#             for step, batch in enumerate(train_dataloader):

#                 if step % 40 == 0 and not step == 0:
#                     print('Batch {:>5,}  of  {:>5,}.'.format(
#                         step, len(train_dataloader)))

#                 self.model.zero_grad()

#                 b_input_ids = batch[0]
#                 b_input_mask = batch[1]
#                 b_labels = batch[2]
#                 loss, logits = self.model(
#                     b_input_ids,
#                     token_type_ids=None,
#                     attention_mask=b_input_mask,
#                     labels=b_labels
#                 )

#                 total_train_loss += loss.item()

#                 loss.backward()

#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

#                 optimizer.step()
#                 scheduler.step()

#             avg_train_loss = total_train_loss / len(train_dataloader)

#             print("")
#             print("  Average training loss: {0:.2f}".format(avg_train_loss))

#     def predict(self, X):
#         """Predict the answer"""
#         pass

#     def evaluate(self, X, y):
#         """Evaluate the model"""
#         pass

#     def save(self, language):
#         self.model.save_pretrained(
#             self.get_save_path(language, 'model')
#         )

#     def load(self, language):
#         self.model.from_pretrained(
#             self.get_save_path(language, 'model')
#         )


# -- Resources --
# https://huggingface.co/Finnish-NLP/gpt2-finnish
# https://www.modeldifferently.com/en/2021/12/generación-de-fake-news-con-gpt-2/
# https://github.com/huggingface/transformers/issues/1528#issuecomment-544977912

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd

base_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
base_tokenizer.pad_token = base_tokenizer.eos_token
base_model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True)


def make_prediction(txt):
    base_model.eval()
    text_ids = base_tokenizer.encode(txt, return_tensors='pt')

    generated_text_samples = base_model.generate(
        text_ids,
        do_sample=True,
        max_length=30,
        num_return_sequences=5
    )

    for i, beam in enumerate(generated_text_samples):
        print(f"{i}: {base_tokenizer.decode(beam, skip_special_tokens=True)}")


make_prediction("What is the")
make_prediction("How do")
make_prediction("When will")

df = pd.read_json('../cleaned_data/english.json', orient='records')
df_train, df_val = train_test_split(df, train_size=0.8, random_state=0)
train_dataset = Dataset.from_pandas(df_train[['question_text']])
val_dataset = Dataset.from_pandas(df_val[['question_text']])

model_headlines_path = './'


def tokenize_function(examples):
    return base_tokenizer(examples['question_text'], padding=True)


tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    remove_columns=['question_text'],
)
tokenized_val_dataset = val_dataset.map(
    tokenize_function,
    remove_columns=['question_text'],
)

training_args = TrainingArguments(
    output_dir=model_headlines_path,          # output directory
    num_train_epochs=6,              # total # of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=200,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir=model_headlines_path,            # directory for storing logs
    prediction_loss_only=True,
    save_steps=10000
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=base_tokenizer,
    mlm=False
)


trainer = Trainer(
    # the instantiated 🤗 Transformers model to be trained
    model=base_model,
    args=training_args,                  # training arguments, defined above
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,         # training dataset
    eval_dataset=tokenized_val_dataset            # evaluation dataset
)
trainer.train()


make_prediction("What is the")
make_prediction("How do")
make_prediction("When will")

trainer.save_model()
base_tokenizer.save_pretrained(model_headlines_path)
