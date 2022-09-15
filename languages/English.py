from nltk import download
from nltk.stem.snowball import SnowballStemmer  
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from .LanguageModel import LanguageModel

class English(LanguageModel):
    def __init__(self):
        super().__init__("english")

    def clean(self, text):
        lower = [x.lower() for x in text]
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer("english")
        words = [stemmer.stem(word) for word in lower if not word in stop_words]
        return words

    def tokenize(self, text):
        return word_tokenize(text)

    