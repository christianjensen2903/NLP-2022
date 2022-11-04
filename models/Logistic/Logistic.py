from models.Model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from pickle import dump, load
import matplotlib.pyplot as plt
import pandas as pd

class Logistic(Model):
    def __init__(self):
        super().__init__()
        self.question_vectorizer = None
        self.plaintext_vectorizer = None
        self.first_word_vectorizer = None
        self.model = LogisticRegression(seed=42)

    def train(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)

    def save(self):
        dump((self.model, self.question_vectorizer, self.plaintext_vectorizer, self.first_word_vectorizer), open(self.get_save_path('pkl'), 'wb'))

    def load(self):
        self.model, self.question_vectorizer, self.plaintext_vectorizer, self.first_word_vectorizer = load(open(self.get_save_path('pkl'), 'rb'))

    def weights(self):
        return dict(zip(
            [f'*QUESTION* {x}' for x in self.question_vectorizer.vocabulary_] +
            [f'*PLAINTEXT* {x}' for x in self.plaintext_vectorizer.vocabulary_] +
            [f'*FIRSTWORD* {x}' for x in self.first_word_vectorizer.vocabulary_] +
            ['*OVERLAP*'] +
            [f'*QUESTION_CONTINOUS* {x}' for x in range(100)] +
            [f'*PLAINTEXT_CONTINOUS* {x}' for x in range(100)] +
            ['*EUCLIDEAN*'] +
            ['*COSINE*'] +
            ['*BERT_SCORE*'],
            list(self.model.coef_[0]),
        ))

    def explainability(self , X , y , n = 10):
        most_important = sorted(self.weights().items(), key=lambda item: abs(item[1]), reverse=True)[:n]
        names , values = list(zip(*most_important))

        plot_df = pd.DataFrame()
        plot_df['names'] = names
        plot_df['values'] = values
        plot_df.plot( x = 'names' , y = 'values', kind='bar' , figsize=(20,10))
        plt.show()

        print (
            "EXPLAINABILITY:\n",
            "Top {} weights for positive:\n".format(n),
            sorted(self.weights().items(), key=lambda item: item[1], reverse=True)[:n],         # n most positive
            "\n\n",
            "Top {} weights for negative:\n".format(n),
            sorted(self.weights().items(), key=lambda item: item[1], reverse=False)[:n],        # n most negative
            "\n\n",
            "Top {} most important weights:\n".format(n),
            most_important,                                                                     # n most important
            "\n\n",
            "Top {} least important weights:\n".format(n),
            sorted(self.weights().items(), key=lambda item: abs(item[1]), reverse=False)[:n]    # n least important
            )
        print(f'Confusion_matrix Matrix:', confusion_matrix(y , self.model.predict(X) , normalize = "all"))