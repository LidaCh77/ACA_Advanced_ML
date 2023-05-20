import numpy as np
from sklearn.datasets import fetch_lfw_people


class TSNE:
    def __init__(self, n_components, perplexity, learning_rate, n_iter):


        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit_transform(self, X):

        n_samples = X.shape[0]        
        pairwise_distances = np.sum(np.square(X), axis=1).reshape((n_samples, 1))
        pairwise_distances -= 2 * np.dot(X,X.T)
        pairwise_distances += np.sum(np.square(X), axis=1).reshape((1, n_samples))

        
        embedding = np.random.randn(n_samples, self.n_components)

        for i in range(self.n_iter):
            # conditional probabilities
            beta = 1 / self.perplexity
            similarities = np.exp(-pairwise_distances * beta)
            np.fill_diagonal(similarities, 0)
            sum_similarities = np.sum(similarities, axis=1).reshape((n_samples, 1))
            p_conditional = similarities / sum_similarities

            # gradients
            p_joint = (p_conditional + p_conditional.T) / (2 * n_samples)
            p_difference = p_joint - 1 / (1 + pairwise_distances)
            q_difference = 1 / (1 + pairwise_distances)
            np.fill_diagonal(q_difference, 0)
            q_difference /= np.sum(q_difference)
            gradients = 4 * np.dot(p_difference.T, q_difference)

            # update embedding
            embedding += self.learning_rate * gradients

        return embedding


#example


lfw_dataset = fetch_lfw_people(min_faces_per_person=100)
X = lfw_dataset.data

tsne = TSNE(n_components = 2, perplexity=30, learning_rate=200, n_iter=1000)




tsne.fit_transform(X)