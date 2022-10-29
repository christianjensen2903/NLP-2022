from extractors.Extractor import Extractor
from extractors.W2V import W2V
from models.Word2Vec import Word2Vec
from pickle import dump, load
import numpy as np


class Continous(Extractor):
    def __init__(self, language, dataset):
        super().__init__(language, dataset)

    def setup(self, train_data):
        self.word2vec = Word2Vec(W2V, self.language)
        self.word2vec.setup(train_data)
        print("yes1")
        try:
            self.word2vec.load()
        except:
            X, _ = self.word2vec.extract(train_data)
            print("yes2")
            self.word2vec.train(X)
            self.word2vec.save()
            print("yes3")

    def unit_length(self, matrix):
        row_norm = np.linalg.norm(matrix, axis=1)
        new_matrix = matrix / row_norm[:, np.newaxis]
        return new_matrix

    def bert_score(self, context_embeddings, question_embeddings):
        context_embeddings = self.unit_length(context_embeddings)
        question_embeddings = self.unit_length(question_embeddings)
        similarity = context_embeddings@question_embeddings.T
        recall = similarity.max(axis=1).sum()/len(context_embeddings)
        precision = similarity.max(axis=0).sum()/len(question_embeddings)
        f1 = 2*recall*precision/(recall+precision)
        return f1

    def run(self, data):
        question_representations = [
            self.word2vec.predict(sentence) for sentence in data['tokenized_question']
        ]
        context_representations = [
            self.word2vec.predict(sentence) for sentence in data['tokenized_plaintext']
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
        bert_scores = np.array([
            self.bert_score(question, context)
            for question, context in zip(question_representations, context_representations)
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
