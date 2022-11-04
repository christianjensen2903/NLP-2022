from Preprocess import Preprocess
from Pipeline import Pipeline
from languages.English import English
import pandas as pd
from models.Logistic.CBOW_BOWLogistic import CBOW_BOWLogistic
from models.Model import Model
from models.RandomForest.CBOW_BOWRandomForest import CBOW_BOWRandomForest
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')



language = English()
pipeline = Pipeline()

# Get the preprocessed data and split it into training and validation data
preprocessor = Preprocess(language.tokenize, language.clean)
preprocessor.from_datasets = False
# raw_data = {
#   "question_text"      : ["When was Queen Elizabeth II born?" , "What is the name for a male horse?" , "What is the name for a male horse?"],
#   "document_plaintext" : [
#     "Queen Elizabeth II was the queen of England from 1952 until she died on the 8th of september 2022." ,
#     "A horse is frequently referred to as a stallion once he fathers a foal. While in most of the western world stallions are primarily kept for breeding, it is popular in parts of the Middle East and Asia for stallions to be used for riding (almost always by men)." ,
#     "A mare is an adult female horse or other equine. In most cases, a mare is a female horse over the age of three, and a filly is a female horse three and younger. In Thoroughbred horse racing, a mare is defined as a female horse more than four years old. The word can also be used for other female equine animals, particularly mules and zebras, but a female donkey is usually called a 'jenny'. A broodmare is a mare used for breeding. A horse's female parent is known as its dam."]
# }

raw_data = {
  "question_text"      : ["When was Queen Elizabeth II born?" , "What is an uncastrated male horse called" , "How can fast does the earth spin around its own axis"],
  "document_plaintext" : [
    "Queen Elizabeth II was the queen of England from 1952 until she died on the 8th of september 2022." ,
    "A horse of masculine gneder which has not been castrated can be referred to as a stallion in American English" ,
    "Earth earth earth earth spin spin spin spin around around around around"]
}

dataset = pd.DataFrame.from_dict(data = raw_data)
data = preprocessor.preprocess(dataset)


models :list[Model] =[
  CBOW_BOWLogistic(),
  CBOW_BOWRandomForest()
]
for model in models:
  model.set_language(language.name)
  model.load()

  X_validation = model.extract_X(data)
  y_validation = [0,1,0]

  pipeline.evaluate(
              model,
              X_validation,
              y_validation
          )

  print(model.predict(X_validation))