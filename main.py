from models.ContinuousBOWLogistic import ContinuousBOWLogistic
from models.ContinuousLogistic import ContinuousLogistic
from languages.LanguageModel import LanguageModel
from models.GPT2Logistic import GPT2Logistic
from DataExploration import DataExploration
from models.BOWLogistic import BOWLogistic
# from languages.Japanese import Japanese
from languages.English import English
from languages.Finnish import Finnish
from Preprocess import Preprocess
from models.Model import Model
from Pipeline import Pipeline
from typing import List
import datasets

# Is used to minimize the clutter in the console
datasets.logging.set_verbosity_error()

# Define the languages to be used
languages: List[LanguageModel] = [
    English(),
    Finnish(),
    # Japanese()
]


bowLogistic, continuousBOWLogistic, continuousLogistic, gpt2Logisitic = BOWLogistic(
), ContinuousBOWLogistic(), ContinuousLogistic(), GPT2Logistic()
# Define the models to be tested
models: List[Model] = [
    gpt2Logisitic,
    bowLogistic,
    continuousBOWLogistic,
    continuousLogistic
]

# Define the parameters to be used in the grid search
parameters = {
    bowLogistic: {
        'penalty': ['l2'],
        'C': [0.1, 1, 10, 100, 1000],
    },
    continuousBOWLogistic: {
        'penalty': ['l2'],
        'C': [0.1, 1, 10, 100, 1000],
    },
    continuousLogistic: {
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
    data_exploration = DataExploration(train_data)
    data_exploration.find_frequent_words()

    # Train and evaluate all the models
    for model in models:
        model.set_language(language.name)
        print(f'\n - Model: {model.__class__.__name__}')
        try:
            model.load()
        except:
            print('Extracting training features...')
            X_train = model.extract_X(train_data)
            y_train = train_data['is_answerable']
            if grid_search:
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
            
        print('Extracting validation features...')
        X_validation = model.extract_X(validation_data)
        y_validation = validation_data['is_answerable']
        pipeline.evaluate(
            model,
            X_validation,
            y_validation
        )
