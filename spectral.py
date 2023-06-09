# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1L_54R1J_SY7xS0TvdhQYZvRJ_jp5ac_i
"""

import numpy as np
import numpy.linalg
from sklearn.cluster import KMeans

class SpectralClustering:

    def __init__(self,data,clusters,sigma):
      self.data = data
      self.clusters = clusters
      self.sigma = sigma

  

    def fit(self):



      #definitions,assuming the solution with normalized ratio cut approach
      


      distance = np.sqrt(np.sum((self.data[:, np.newaxis] - self.data[np.newaxis, :])**2, axis=-1))
      W_matrix = np.exp(-distance ** 2 / (2 * self.sigma ** 2))
      D_matrix = np.diag(np.sum(W_matrix, axis=1))
      Laplasian = D_matrix - W_matrix
      norm_Laplasian = np.linalg.inv(D_matrix) @ Laplasian



      # eigenvalues/vectors of the normalized Laplacian 
      eigenvalues, eigenvectors = np.linalg.eig(norm_Laplasian)

      # Sort the eigenvectors according to their eigenvalues in ascending order

      sorted_values = np.argsort(eigenvalues)
      sorted_vectors = eigenvectors[:, sorted_values]




      # Use the eigenvectors corresponding to the smallest eigenvalues to form the embedding matrix
      embedding = sorted_vectors[:, 1:self.clusters+1]

      # Normalize the rows of the embedding matrix
      norm_embedding = embedding / np.sqrt(np.sum(embedding**2, axis = 1, keepdims = True))

      # Use k-means clustering to cluster the data points in the low-dimensional space
      kmeans = KMeans(n_clusters= self.clusters)
      kmeans.fit(norm_embedding)

      return kmeans.lebels_
