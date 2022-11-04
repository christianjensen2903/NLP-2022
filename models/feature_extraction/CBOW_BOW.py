from models.feature_extraction.BOW import BOW
from gensim.models import Word2Vec as GensimWord2Vec
from models.Word2Vec import Word2Vec
import numpy as np


class CBOW_BOW(BOW):
    def __init__(self):
        super().__init__()
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

    def unit_length(self, matrix):
        row_norm = np.linalg.norm(matrix, axis=1)
        # if row_norm.any() == 0:
            # print(row_norm)
            # row_norm = np.array([0.1])
        new_matrix = matrix / row_norm[:, np.newaxis]
        return np.nan_to_num(new_matrix)

    def bert_score(self, context_embeddings, question_embeddings):
        context_embeddings = self.unit_length(context_embeddings)
        question_embeddings = self.unit_length(question_embeddings)
        similarity = context_embeddings@question_embeddings.T
        recall = similarity.max(axis=1).sum()/len(context_embeddings)
        precision = similarity.max(axis=0).sum()/len(question_embeddings)
        f1 = 2*recall*precision/(recall+precision)
        return f1

    def get_continuous_representation(self, dataset):
        if self.word2vec is None:
            self.word2vec = self.get_word2vec(dataset)
        question_representations = [
            self.word2vec.predict(sentence) for sentence in dataset['tokenized_question']
        ]
        context_representations = [
            self.word2vec.predict(sentence) for sentence in dataset['tokenized_plaintext']
        ]
        question_mean_representation = np.array([
            repr.mean(axis=0) for repr in question_representations
        ])
        context_mean_representation = np.array([
            repr.mean(axis=0) for repr in context_representations
        ])
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
        bert_scores = np.nan_to_num(
            np.array([
                self.bert_score(question, context)
                for question, context in zip(question_representations, context_representations)
            ]).reshape(-1, 1)
        )
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
