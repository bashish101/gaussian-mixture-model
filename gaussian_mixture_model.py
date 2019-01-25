import numpy as np

class GMM(object):
	def __init__(self,
		     num_states = 3,
		     tolerence = 1E-4,
		     max_iterations = 100):
		self.num_states = num_states
		self.tolerence = tolerence
		self.max_iterations = max_iterations

	def gaussian_pdf(self, x, mu, sigma):
		epsilon = np.finfo(float).eps
		p = len(x)
		centered = np.matrix(x - mu)
		norm_factor = (((2 * np.pi) ** p) * np.absolute(np.linalg.det(sigma)) + epsilon) ** 0.5
		pb = (1. / norm_factor) * np.exp(- 0.5 * centered * np.linalg.inv(sigma) * centered.T)
		return np.squeeze(np.array(pb))

	def compute_likelihood(self, x, num_states, mu, sigma, pi):
		score = 0.0
		for idx in range(num_states):
			score += pi[idx] * self.gaussian_pdf(x, mu[idx], sigma[idx])
		return score

	def compute_log_likelihood(self, data, num_states, mu, sigma, pi):
		ll = 0.0
		for idx in range(len(data)):
			ll += np.log(self.compute_likelihood(data[idx], num_states, mu, sigma, pi))
		return ll

	def display_progress(self, progress, msg = None):
		completed = '#' * progress
		remaining = ' ' * (100 - progress)
		
		print ('\r[{0}{1}] {2}% | {3}'.format(completed, remaining, progress, msg), end = '\r')

	def k_means(self, X, k, max_iters = 30):
		centroids = np.array(X[np.random.choice(np.arange(len(X)), k, replace = False), :])
		for idx in range(max_iters):
			centroids_indices = np.array([np.argmin([np.dot(x - y, x - y) for y in centroids]) for x in X])
			centroids = np.array([np.mean(X[centroids_indices == centroid_idx], axis = 0) for centroid_idx in range(k)])
		return np.array(centroids), centroids_indices

	def em_init_params(self, data, num_states):
		size, num_feats = np.shape(data)[:2]

		pi = np.empty((num_states,), dtype = float)
		sigma = np.empty((num_states, num_feats, num_feats), dtype = float)

		centroids, centroid_indices = self.k_means(data, num_states)

		mu = centroids

		for idx in range(num_states):
			data_indices = np.where(centroid_indices == idx)
			pi[idx] = len(data_indices)
			sigma[idx] = np.cov(data.T) + 1E-6 * np.eye(num_feats)
		pi = pi / size
		return (mu, sigma, pi)

	def em(self, data, num_states, mu, sigma, pi, tolerence = 1E-3, max_iters = 100):
		size, num_feats = data.shape[:2]

		prev_ll = self.compute_log_likelihood(data, num_states, mu, sigma, pi)

		converged = False
		for iter_idx in range(max_iters):
			# Start Expectation 
			r = np.zeros((size, num_states))
			for idx1 in range(size):
				for idx2 in range(num_states):
					prior = pi[idx2]
					pb = self.gaussian_pdf(data[idx1], mu[idx2], sigma[idx2]) 
					l = self.compute_likelihood(data[idx1], num_states, mu, sigma, pi)
					r[idx1][idx2] = prior * pb / l
			# End Expectation			

			# Start Maximization 
			mu[:] = 0.		# np.zeros((num_states, num_feats))
			sigma[:] = 0.		# np.zeros((num_states, num_feats, num_feats))
			pi[:] = 0.		# np.zeros((num_states))

			marg_r = np.zeros((num_states,))
			for idx1 in range(num_states):
				for idx2 in range(size):
					marg_r[idx1] += r[idx2][idx1]
					mu[idx1] += r[idx2][idx1] * data[idx2]
				mu[idx1] /= marg_r[idx1]

				for idx2 in range(size):
					centered = np.zeros((1, num_feats)) + data[idx2] - mu[idx1]
					sigma[idx1] += (r[idx2][idx1] / marg_r[idx1]) * centered * centered.T
				pi[idx1] = marg_r[idx1] / size			
			# End Maximization

			ll = self.compute_log_likelihood(data, num_states, mu, sigma, pi)

			progress = int((iter_idx / max_iters) * 100)
			msg = "Current Log likelihood: {:.4f}".format(ll)
			self.display_progress(progress, msg)

			if abs(ll - prev_ll) < tolerence:
				break

			prev_ll = ll
			
		return mu, sigma, pi, r

	def fit(self, data):
		mu, sigma, pi = self.em_init_params(data, self.num_states)
 
		self.mu, self.sigma, self.pi, self.r = self.em(data, self.num_states, mu, sigma, pi, self.tolerence, self.max_iterations)

	def predict(self, data):
		size = len(data)
		predictions = []
		dist_arr = np.empty(self.num_states, dtype = float)
		sigma_inv = [np.linalg.inv(cov) for cov in self.sigma]

		for idx1 in range(size):
			x = data[idx1]
			for idx2 in range(self.num_states):
				dist_arr[idx2] = np.dot(np.dot((x - self.mu[idx2]).T, sigma_inv[idx2]), x - self.mu[idx2])
			predictions.append(np.argmin(dist_arr))

		return np.array(predictions)



