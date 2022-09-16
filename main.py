from models.BOWLogistic import BOWLogistic
from languages.LanguageModel import LanguageModel
from DataExploration import DataExploration
# from languages.Japanese import Japanese
from languages.English import English
from languages.Finnish import Finnish
from Preprocess import Preprocess
from Pipeline import Pipeline
from models.Word2Vec import Word2Vec
from models.Model import Model
from typing import List
import datasets

# Is used to minimize the clutter in the console
datasets.logging.set_verbosity_error()


# Define the languages to be used
languages: List[LanguageModel] = [English(), Finnish()]
# Define the models to be tested
models: List[Model] = [BOWLogistic()]


# Run trough the pipeline for all languages and models
for language in languages:
    print(f'-- Language: {language.name} --')
    pipeline = Pipeline()

    # Get the preprocessed data and split it into training and validation data
    preprocessor = Preprocess(language.tokenize, language.clean)
    data = pipeline.get_data(language=language.name, preproccesor=preprocessor)
    train_data, validation_data = pipeline.split_data(data)
    model = Word2Vec()
    X_train = model.extract_X(train_data)
    try:
        model.load(language)
    except:
        model.train(X_train)
        model.save(language)


    print(model.predict(['What', 'is', 'hello']))
    print(model.predict(['What', 'is', 'hello']).mean(axis=0))
    # # Explore the data
    # data_exploration = DataExploration(train_data)
    # data_exploration.find_frequent_words()

    # # Train and evaluate all the models
    # for model in models:
    #     print('\nExtracting features...')
    #     X_train = model.extract_X(train_data)
    #     y_train = train_data['is_answerable']
    #     X_validation = model.extract_X(validation_data)
    #     y_validation = validation_data['is_answerable']
    #     try:
    #         model.load(language)
    #     except:
    #         model = pipeline.train(model, X_train, y_train)
    #         model.save(language)
    #     pipeline.evaluate(model, X_validation, y_validation)
