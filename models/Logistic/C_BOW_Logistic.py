from models.Model import Model
from sklearn.feature_extraction.text import CountVectorizer
from models.feature import get_vectorizers
from sklearn.linear_model import LogisticRegression
from models.Word2Vec import Word2Vec
from pickle import dump, load


class C_BOW_Logistic(Model):
    def __init__(self, language, config):
        super().__init__(language, config)
        self.model = LogisticRegression()

    def get_word2vec(dataset):
        word2vec = Word2Vec(self.language, {})
        try:
            word2vec.load()
        except:
            word2vec.train(word2vec.extract_X(dataset))
            word2vec.save()
        return word2vec

    def setup(self, dataset):
        def identity_function(x):
            return x
        self.plaintext_vectorizer = CountVectorizer(
            tokenizer=identity_function,  # Avoid tokenizing again
            preprocessor=identity_function
        ).fit(
            dataset['tokenized_plaintext']
        )
        self.question_vectorizer = CountVectorizer(
            tokenizer=identity_function,
            preprocessor=identity_function
        ).fit(
            dataset['tokenized_question']
        )
        self.first_word_vectorizer = CountVectorizer(
            tokenizer=identity_function,
            preprocessor=identity_function
        ).fit(
            dataset['tokenized_question'].str[0]
        )
        self.word2vec = get_word2vec()

    def extract_X(self, dataset):
        question_representations = [
            word2vec.predict(sentence) for sentence in dataset['tokenized_question']
        ]
        context_representations = [
            word2vec.predict(sentence) for sentence in dataset['tokenized_plaintext']
        ]
        question_mean_representation = np.array([
            repr.mean(axis=0) for repr in question_representations
        ])
        context_mean_representation = np.array([
            repr.mean(axis=0) for repr in context_representations
        ])
        bert_scores = np.array([
            bert_score(context, question)
            for context, question in zip(context_representations, question_representations)
        ]).reshape(-1, 1)
        distance_between_representations = np.linalg.norm(
            question_mean_representation - context_mean_representation,
            axis=1
        ).reshape(-1, 1)
        cosine_similarity = np.array([
            np.dot(question_mean_representation[i], context_mean_representation[i]) / (
                np.linalg.norm(
                    question_mean_representation[i]) * np.linalg.norm(context_mean_representation[i])
            ) for i in range(len(question_mean_representation))
        ]).reshape(-1, 1)
        return np.concatenate(
            (
                question_mean_representation,
                context_mean_representation,
                distance_between_representations,
                cosine_similarity,
                bert_scores
            ),
            axis=1
        )

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
