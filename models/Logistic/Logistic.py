from models.Model import Model
from sklearn.linear_model import LogisticRegression
from pickle import dump, load


class Logistic(Model):
    def __init__(self):
        super().__init__()
        self.question_vectorizer = None
        self.plaintext_vectorizer = None
        self.first_word_vectorizer = None
        self.model = LogisticRegression()

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
            list(map(lambda x: "*QUESTION* " + x, [*self.question_vectorizer.vocabulary_])) +
            list(map(lambda x: "*PLAINTEXT* " + x, [*self.plaintext_vectorizer.vocabulary_])) +
            list(map(lambda x: "*FIRSTWORD* " + x, [*self.first_word_vectorizer.vocabulary_])) +
            ['*OVERLAP*'],
            list(self.model.coef_[0]),
        ))

    def explainability(self , n = 5):
        print (
            "EXPLAINABILITY:\n",
            "Top {} weights for positive:\n".format(n),
            sorted(self.weights().items(), key=lambda item: item[1], reverse=True)[:n], # n most positive
            "\n\n",
            "Top {} weights for negative:\n".format(n),
            sorted(self.weights().items(), key=lambda item: item[1], reverse=False)[:n] # n most negative
            )