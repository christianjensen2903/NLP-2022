from models.Model import Model
from sklearn.neural_network import MLPClassifier
from pickle import dump, load


class MLP(Model):
    def __init__(self, language):
        super().__init__(language)
        self.question_vectorizer = None
        self.plaintext_vectorizer = None
        self.first_word_vectorizer = None
        self.model = MLPClassifier(early_stopping=True, hidden_layer_sizes=(50, 50))

    def train(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)

    def save(self):
        dump((self.model, self.question_vectorizer, self.plaintext_vectorizer,
             self.first_word_vectorizer), open(self.get_save_path('pkl'), 'wb'))

    def load(self):
        self.model, self.question_vectorizer, self.plaintext_vectorizer, self.first_word_vectorizer = load(
            open(self.get_save_path('pkl'), 'rb'))

    def weights(self):
        pass

    def explainability(self, n=5):
        pass
