from Preprocess import Preprocess
from Pipeline import Pipeline
from languages.English import English
from models.SequenceLabeller import SequenceLabeller

english = English()

preproccesor = Preprocess(english.tokenize, english.clean)
data = Pipeline().get_data(language=english.name, preproccesor=preproccesor)

sequence_labeller = SequenceLabeller()

tagged_data = sequence_labeller.convert_to_iob(data)

print(tagged_data)
tagged_data.to_csv('tagged_data.csv', index=False)
