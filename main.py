from models.BOWLogistic import BOWLogistic
from models.ContinuousBOWLogistic import ContinuousBOWLogistic
from models.ContinuousLogistic import ContinuousLogistic
from models.BertClassifier import BertClassifier
from languages.LanguageModel import LanguageModel
from DataExploration import DataExploration
from languages.English import English
from languages.Finnish import Finnish
# from languages.Japanese import Japanese
from Preprocess import Preprocess
from Pipeline import Pipeline
from models.Word2Vec import Word2Vec
from models.Model import Model
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


bowLogistic, continuousBOWLogistic, continuousLogistic, word2Vec, bertClassifier = BOWLogistic(
), ContinuousBOWLogistic(), ContinuousLogistic(), Word2Vec(), BertClassifier()
# Define the models to be tested
models: List[Model] = [
    bertClassifier,
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
        print(f'\n - Model: {model.__class__.__name__}')
        print('Extracting features...')
        try:  # Added to make sure word2vec is trained for each language
            model.word2vec = None
        except:
            pass
        X_train = model.extract_X(train_data, language)
        y_train = train_data['is_answerable']
        X_validation = model.extract_X(validation_data, language)
        y_validation = validation_data['is_answerable']
        try:
            model.load(language.name)
        except:
            if grid_search:
                model = pipeline.grid_search(
                    model, X_train, y_train, parameters[model])
            else:
                model = pipeline.train(model, X_train, y_train)
            model.save(language.name)
        pipeline.evaluate(model, X_validation, y_validation)
