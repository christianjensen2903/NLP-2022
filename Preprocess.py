from nltk import download
from nltk.stem.snowball import SnowballStemmer  
from nltk.corpus import stopwords
import advertools as adv


#Downloading stopwords
download('stopwords')
download('punkt')

# Defining cleaning functions
def clean_english(text):
    lower = [x.lower() for x in text]
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    words = [stemmer.stem(word) for word in lower if not word in stop_words]
    return words

def clean_finnish(text):
    lower = [x.lower() for x in text]
    stop_words = set(stopwords.words('finnish'))
    stemmer = SnowballStemmer("finnish")
    words = [stemmer.stem(word) for word in lower if not word in stop_words]
    return words

# def clean_japanese(text):
    # lower = [x.lower() for x in text]
    # stop_words = set(adv.stopwords['japanese'])
    # words = [word for word in lower if not word in stop_words]
    # return words


# Defining tagger for japanese tokenizer
from fugashi import Tagger
japanese_tagger = Tagger('-Owakati') # Tagger has initial startup overhead, therefore it is defined here and not in lambda function
tokenize_japanese = lambda text : japanese_tagger.parse(text).split(" ")
# tokenize_japanese = lambda text : text
