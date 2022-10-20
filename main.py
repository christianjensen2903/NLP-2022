from models.Model import Model
from models.GPT2CBOWLogistic import GPT2CBOWLogistic
from models.GPT2Generator import GPT2Generator
from models.Logistic.BOWLogistic import BOWLogistic
from models.MLP.BOWMLP import BOWMLP
from models.MLP.CBOW_BOWMLP import CBOW_BOWMLP
from models.MLP.CBOWMLP import CBOWMLP
from models.Logistic.CBOW_BOWLogistic import CBOW_BOWLogistic
from models.Logistic.CBOWLogistic import CBOWLogistic
from models.XGBoost.BOWXGBoost import BOWXGBoost
from models.XGBoost.CBOW_BOWXGBoost import CBOW_BOWXGBoost
from models.XGBoost.CBOWXGBoost import CBOWXGBoost

from languages.LanguageModel import LanguageModel
from DataExploration import DataExploration
# from languages.Japanese import Japanese
from languages.English import English
from languages.Finnish import Finnish
from Preprocess import Preprocess
from Pipeline import Pipeline
from typing import List
import torch
import datasets

# Is used to minimize the clutter in the console
datasets.logging.set_verbosity_error()

# Define the languages to be used
languages: List[LanguageModel] = [
    English(),
    Finnish(),
    # Japanese()
]

# gpt2Generator = GPT2Generator()
torch.cuda.empty_cache()

# gpt2CBOWLogistic = GPT2CBOWLogistic()
bowMLP = BOWMLP()
cbow_BOWMLP = CBOW_BOWMLP()
cbowMLP = CBOWMLP()
bowLogistic = BOWLogistic()
cBOWLogistic = CBOWLogistic()
cBOW_BOWLogistic = CBOW_BOWLogistic()
BOW_XGb = BOWXGBoost()
cBOW_BOWXGBoost = CBOW_BOWXGBoost()
cBOWXGBoost = CBOWXGBoost()

# Define the models to be tested
models: List[Model] = [
    # gpt2Generator,
    # gpt2CBOWLogistic,
    bowMLP,
    cbow_BOWMLP,
    cbowMLP,
    bowLogistic,
    cBOW_BOWLogistic,
    cBOWLogistic,
    # BOW_XGb,
    # cBOW_BOWXGBoost,
    # cBOWXGBoost,
]

question_beginning = {
    'english': ['Question: When', 'Question: What', 'Question: How'],
    'finnish': ['Question: Milloin', 'Question: Mikä', 'Question: Missä'],
    'japanese': ['Question: 日本', 'Question: 『', 'Question: アメリカ']
}

# Define the parameters to be used in the grid search
parameters = {
    bowLogistic: {
        'penalty': ['l2'],
        'C': [0.1, 1, 10, 100, 1000],
    },
    cBOW_BOWLogistic: {
        'penalty': ['l2'],
        'C': [0.1, 1, 10, 100, 1000],
    },
    cBOWLogistic: {
        'penalty': ['l2'],
        'C': [0.1, 1, 10, 100, 1000],
    }
}

grid_search = False

# Run trough the pipeline for all languages and models
for language in languages:
    print(f'\n\n--- Language: {language.name} ---')
    pipeline = Pipeline()

    # Get the preprocessed data and split it into training and validation data
    preprocessor = Preprocess(language.tokenize, language.clean)
    data = pipeline.get_data(language=language.name, preproccesor=preprocessor)
    train_data, validation_data = pipeline.split_data(data)

    # Explore the data
    # data_exploration = DataExploration(train_data)
    # data_exploration.find_frequent_words()

    # Train and evaluate all the models
    for model in models:
        model_name = model.__class__.__name__
        model.set_language(language.name)
        print(f'\n - Model: {model_name}')

        print('Extracting features...')
        X_train = model.extract_X(train_data)
        y_train = train_data['is_answerable']
        X_validation = model.extract_X(validation_data)
        y_validation = validation_data['is_answerable']

        try:
            model.load()
            if model_name == "GPT2Generator":
                for starting_word in question_beginning[language.name]:
                    model.generate_text(starting_word)
                model.get_perplexity(X_validation)
        except:
            if grid_search:
                pass
                model = pipeline.grid_search(
                    model,
                    X_train,
                    y_train,
                    parameters[model]
                )
            else:
                model = pipeline.train(
                    model,
                    X_train,
                    y_train
                )
            model.save()
        pipeline.evaluate(
            model,
            X_validation,
            y_validation
        )
        model.explainability()
