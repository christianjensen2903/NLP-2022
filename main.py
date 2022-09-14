import numpy as np
from Pipeline import Pipeline
from Language import Language
from BinaryQuestionClassifier import BinaryQuestionClassifier
from Preprocess import clean_english , clean_finnish , clean_japanese , tokenize_japanese
from datasets import load_dataset
from nltk.tokenize import word_tokenize



dataset = load_dataset('copenlu/answerable_tydiqa')


get_data = lambda language: dataset.filter(lambda x: x['language'] == language)


# TASK 1.1a
# Preprocess the data


# Fetching language appropriate pipelines and getting data
languages = {
    'english': Language(
        name = 'english',
        tokenizer = word_tokenize,
        cleaner = clean_english,
        pipeline = Pipeline(get_data('english'))
        ),
    'japanese': Language(
        name = 'japanese',
        tokenizer = tokenize_japanese,
        cleaner = clean_japanese,
        pipeline = Pipeline(get_data('japanese'))
        ),
    'finnish': Language(
        name = 'finnish',
        tokenizer = word_tokenize,
        cleaner = clean_finnish,
        pipeline = Pipeline(get_data('finnish'))
        ),
}

# Preprocessing
for language in languages.values():
    language.pipeline.tokenize(language.tokenizer)
    language.pipeline.clean(language.cleaner)
    language.pipeline.label_answerable()



#TASK 1.1b

# Find the most common first and last words in each language
for language in languages.values():
    count_words = lambda text: np.unique(text, return_counts=True) # Count occurences of words in text
    sort_words = lambda word_count: np.argsort(word_count[1])[::-1] # Get list of sorted indices based on most frequent words
    zip_words = lambda word_counts, sort_indices: list(zip(word_counts[0][sort_indices],word_counts[1][sort_indices])) # Zip the most frequent words with its number of occurences
    def find_most_common(text):
        """Finds the most frequent words in a text together with its number of occurences"""
        word_count = count_words(text)
        return zip_words(word_count, sort_words(word_count))


    tokenized_questions = language.pipeline.train_data['tokenized_question']
    first = [sublist[0] for sublist in tokenized_questions]
    last = [sublist[-1] for sublist in tokenized_questions]
    
    print(f"""
    Language: {language.name}
    Most frequent first words:
    {find_most_common(first)[:5]}
    Most frequent last words:
    {find_most_common(last)[:5]}
    """)


#Task 1.2
for language in languages.values():
    print(f'\nLanguage: {language.name}')
    model = BinaryQuestionClassifier()
    X = model.extract_X(language.pipeline.train_data)
    y = language.pipeline.train_data['is_answerable']
    model.train(X, y)

    X_val = model.extract_X(language.pipeline.validation_data)
    y_val = language.pipeline.validation_data['is_answerable']
    model.evaluate(X_val, y_val)