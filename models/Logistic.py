from models.Model import Model
from sklearn.linear_model import LogisticRegression
from pickle import dump, load


class Logistic(Model):
    def __init__(self, extractor, language, config={}):
        super().__init__(extractor, language, config)
        self.model = LogisticRegression()

    def setup(self, train_data):
        # Initialize extractor
        self.extractor = self.extractor(
            self.language, train_data
        )

    def train(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)

    def save(self):
        dump(self.model, open(self.get_save_path('pkl'), 'wb'))

    def load(self):
        self.model = load(
            open(self.get_save_path('pkl'), 'rb')
        )

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

    def explainability(self, n=5):
        print(
            "EXPLAINABILITY:\n",
            "Top {} weights for positive:\n".format(n),
            sorted(self.weights().items(), key=lambda item: item[1], reverse=True)[
                :n],  # n most positive
            "\n\n",
            "Top {} weights for negative:\n".format(n),
            sorted(self.weights().items(), key=lambda item: item[1], reverse=False)[
                :n]  # n most negative
        )
