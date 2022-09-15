from Pipeline import Pipeline
from languages.English import English
from languages.Finnish import Finnish
# from languages.Japanese import Japanese
from Preprocess import Preprocess
from DataExploration import DataExploration
from Model import Model
from BinaryQuestionClassifier import BinaryQuestionClassifier
import datasets
from typing import List

from languages.LanguageModel import LanguageModel


datasets.logging.set_verbosity_error() # Is used to minimize the clutter in the console


languages: List[LanguageModel] = [English(), Finnish()] # Define the languages to be used
models: List[Model] = [BinaryQuestionClassifier()] # Define the models to be tested


# Run trough the pipeline for all languages and models
for language in languages:
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
        X_train = model.extract_X(train_data)
        y_train = train_data['is_answerable']
        X_validation = model.extract_X(validation_data)
        y_validation = validation_data['is_answerable']

        model = pipeline.train(model, X_train, y_train)
        pipeline.evaluate(model, X_validation, y_validation)