import numpy as np


class DataExploration():
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def get_data_shape(self):
        return self.data.shape

    def get_data_columns(self):
        return self.data.columns
    
    def find_frequent_words(self):
        """Find the most common first and last words"""
        count_words = lambda text: np.unique(text, return_counts=True) # Count occurences of words in text
        sort_words = lambda word_count: np.argsort(word_count[1])[::-1] # Get list of sorted indices based on most frequent words
        zip_words = lambda word_counts, sort_indices: list(zip(word_counts[0][sort_indices],word_counts[1][sort_indices])) # Zip the most frequent words with its number of occurences
        def find_most_common(text):
            """Finds the most frequent words in a text together with its number of occurences"""
            word_count = count_words(text)
            return zip_words(word_count, sort_words(word_count))


        tokenized_questions = self.data['tokenized_question']
        first = [sublist[0] for sublist in tokenized_questions]
        last = [sublist[-1] for sublist in tokenized_questions]
        
        print(f"""
        Most frequent first words:
        {find_most_common(first)[:5]}
        Most frequent last words:
        {find_most_common(last)[:5]}
        """)