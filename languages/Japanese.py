from .LanguageModel import LanguageModel
from nltk import download
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import advertools as adv
from fugashi import Tagger


class Japanese(LanguageModel):
    def __init__(self):
        super().__init__("japanese")
        self.japanese_tagger = Tagger('-Owakati')

    def clean(self, text):
        lower = [x.lower() for x in text]
        stop_words = set(adv.stopwords['japanese'])
        words = [word for word in lower if not word in stop_words]
        return words

    def tokenize(self, text):
        return self.japanese_tagger.parse(text).split(" ")
