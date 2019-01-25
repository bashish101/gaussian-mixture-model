import numpy as np

class PCA(object):
	def __init__(self, num_components):
		self.num_components = num_components
	
	def calc_covariance_matrix(self, X, Y):
		N = len(X)
		cov_mat = (1 / (N - 1)) * np.dot((X - np.mean(X, axis = 0)).T,  Y - np.mean(Y, axis = 0))
		return cov_mat

	def transform(self, X):
		cov_mat = self.calc_covariance_matrix(X, X)
		eig_val, eig_vec = np.linalg.eig(cov_mat)

		select_idx = eig_val.argsort()[::-1]
		eig_val = eig_val[select_idx][:self.num_components]
		eig_vec = np.atleast_1d(eig_vec[:, select_idx])[:, :self.num_components]

		X_transformed = np.dot(X, eig_vec)
		return X_transformed

