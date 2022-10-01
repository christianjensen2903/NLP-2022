from models.Logistic.BOWLogistic import BOWLogistic
from Preprocess import Preprocess
from Pipeline import Pipeline
from languages.English import English
import pandas as pd
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')



language = English()
pipeline = Pipeline()

# Get the preprocessed data and split it into training and validation data
preprocessor = Preprocess(language.tokenize, language.clean)
preprocessor.from_datasets = False
raw_data = {
  "question_text"      : ["When was Queen Elizabeth II born?" , "What is the name for a male horse?"],
  "document_plaintext" : [
    "Queen Elizabeth II was the queen of England from 1952 until she died on the 8th of september 2022." ,
    "A horse is frequently referred to as a stallion once he fathers a foal. While in most of the western world stallions are primarily kept for breeding, it is popular in parts of the Middle East and Asia for stallions to be used for riding (almost always by men)." ]
}
dataset = pd.DataFrame.from_dict(data = raw_data)
data = preprocessor.preprocess(dataset)


model = BOWLogistic()
model.set_language(language.name)
model.load()


print(data)

X_validation = model.extract_X(data)
y_validation = [0,1]

pipeline.evaluate(
            model,
            X_validation,
            y_validation
        )

print(model.predict(X_validation))