from models.feature_extraction.BOW import BOW
from gensim.models import Word2Vec as GensimWord2Vec
from models.Word2Vec import Word2Vec
import numpy as np


class CBOW_BOW(BOW):
    def __init__(self, language):
        super().__init__(language)
        self.word2vec = None

    def set_language(self, language: str):
        super().set_language(language)

        # Check if word2vec is loaded with the correct language
        if self.word2vec and self.word2vec.language != self.language:
            self.word2vec = None

    def get_word2vec(self, dataset):
        word2vec = Word2Vec()
        word2vec.set_language(self.language)
        try:
            word2vec.load()
        except:
            word2vec.train(word2vec.extract_X(dataset))
            word2vec.save()
        return word2vec

    def get_continuous_representation(self, dataset):
        if self.word2vec is None:
            self.word2vec = self.get_word2vec(dataset)
        question_mean_representation = np.array([
            self.word2vec.predict(sentence).mean(axis=0) for sentence in dataset['tokenized_question']
        ])
        plaintext_mean_representation = np.array([
            self.word2vec.predict(sentence).mean(axis=0) for sentence in dataset['tokenized_plaintext']
        ])
        distance_between_representations = np.linalg.norm(
            question_mean_representation - plaintext_mean_representation,
            axis=1
        ).reshape(-1, 1)
        cosine_similarity = np.array([
            np.dot(question_mean_representation[i], plaintext_mean_representation[i]) / (
                np.linalg.norm(
                    question_mean_representation[i]) * np.linalg.norm(plaintext_mean_representation[i])
            ) for i in range(len(question_mean_representation))
        ]).reshape(-1, 1)
        return np.concatenate(
            (
                question_mean_representation,
                plaintext_mean_representation,
                distance_between_representations,
                cosine_similarity
            ),
            axis=1
        )

    def extract_X(self, dataset):
        bow = super().extract_X(dataset)
        continuous_representation = self.get_continuous_representation(dataset)
        return np.concatenate(
            (
                bow,
                continuous_representation
            ),
            axis=1
        )
