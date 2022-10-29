
from models.Word2Vec import Word2Vec
import numpy as np


def unit_length(matrix):
    row_norm = np.linalg.norm(matrix, axis=1)
    new_matrix = matrix / row_norm[:, np.newaxis]
    return new_matrix


def bert_score(context_embeddings, question_embeddings):
    context_embeddings = unit_length(context_embeddings)
    question_embeddings = unit_length(question_embeddings)
    similarity = context_embeddings@question_embeddings.T
    recall = similarity.max(axis=1).sum()/len(context_embeddings)
    precision = similarity.max(axis=0).sum()/len(question_embeddings)
    f1 = 2*recall*precision/(recall+precision)
    return f1


def get_continuous_representation(dataset, word2vec):
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

def get_bow_representation(dataset, plaintext_vectorizer, question_vectorizer, first_word_vectorizer):
    question_bow = question_vectorizer.transform(
        dataset['tokenized_question']
    )
    plaintext_bow = plaintext_vectorizer.transform(
        dataset['tokenized_plaintext']
    )
    first_word_bow = first_word_vectorizer.transform(
        dataset['tokenized_question'].str[0]
    )
    overlap = np.array([len([e for e in question if e in plaintext]) for question, plaintext in zip(
        dataset['cleaned_question'], dataset['cleaned_plaintext'])])
    X = np.concatenate(
        (
            question_bow.toarray(),
            plaintext_bow.toarray(),
            first_word_bow.toarray(),  # First word in question
            overlap.reshape(-1,  1),  # Amount of words overlapping
        ),
        axis=1
    )
    return X


def get_continous_bow(dataset, word2vec, plaintext_vectorizer, question_vectorizer, first_word_vectorizer):
    return np.concatenate(
        (
            get_continuous_representation(dataset, word2vec),
            get_bow_representation(
                dataset, plaintext_vectorizer, question_vectorizer, first_word_vectorizer)
        ),
        axis=1
    )
