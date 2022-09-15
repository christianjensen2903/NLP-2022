import numpy as np
from Pipeline import Pipeline
from languages.English import English
from languages.Finnish import Finnish
# from languages.Japanese import Japanese
from Preprocess import Preprocess
from DataExploration import DataExploration
# from BinaryQuestionClassifier import BinaryQuestionClassifier
import datasets

# TASK 1.1a
# Preprocess the data

datasets.logging.set_verbosity_error()

# Fetching language appropriate pipelines and getting data
english_pipeline = English()
languages = [English(), Finnish()]
#TASK 1.1b

# Find the most common first and last words in each language
for language in languages:
    pipeline = Pipeline()
    
    preprocessor = Preprocess(language.tokenize, language.clean)
    data = pipeline.get_data(language=language.name, preproccesor=preprocessor)
    train_data, test_data = pipeline.split_data(data)

    data_exploration = DataExploration(train_data)
    data_exploration.find_frequent_words()

    


# #Task 1.2
# for language in languages.values():
#     print(f'\nLanguage: {language.name}')
#     model = BinaryQuestionClassifier()
#     X = model.extract_X(language.pipeline.train_data)
#     y = language.pipeline.train_data['is_answerable']
#     model.train(X, y)

#     X_val = model.extract_X(language.pipeline.validation_data)
#     y_val = language.pipeline.validation_data['is_answerable']
#     model.evaluate(X_val, y_val)