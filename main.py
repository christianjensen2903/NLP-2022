from languages.LanguageModel import LanguageModel
# from languages.Japanese import Japanese
from extractors.BOW import BOW
from extractors.Continous import Continous
from models.Logistic import Logistic
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

torch.cuda.empty_cache()

# Run trough the pipeline for all languages and models
for language in languages:
    print(f'\n\n--- Language: {language.name} ---')
    # Define the models to be tested
    models = [
        Logistic(Continous, language.name),
        Logistic(BOW, language.name)
    ]
    
    pipeline = Pipeline()

    # Get the preprocessed data and split it into training and validation data
    preprocessor = Preprocess(language.tokenize, language.clean)
    data = pipeline.get_data(language=language.name, preproccesor=preprocessor)
    train_data, validation_data = pipeline.split_data(data)
    train_data = train_data.head(100)

    # Train and evaluate all the models
    for model in models:
        print(f'\n - Model: {model.name}')

        print('Setup...')
        model.setup(train_data)

        try:
            model.load()
        except:
            print('Extracting...')
            X_train, y_train = model.extract(train_data)
            model = pipeline.train(
                model,
                X_train,
                y_train
            )
            model.save()

        X_validation, y_validation = model.extract(validation_data)
        pipeline.evaluate(
            model,
            X_validation,
            y_validation
        )
