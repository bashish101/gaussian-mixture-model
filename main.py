import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

from utils import PCA
from gaussian_mixture_model import GMM

def plot_results(X, y, title = None):
	pca = PCA(num_components = 2)

	X_transformed = pca.transform(X)
	x_coords = X_transformed[:, 0]
	y_coords = X_transformed[:, 1]

	y = np.array(y).astype(int)
	classes = np.unique(y)

	cmap = plt.get_cmap('viridis')
	colors = [cmap(val) for val in np.linspace(0, 1, len(classes))]

	for idx, cls in enumerate(classes):
		x_coord = x_coords[y == cls]
		y_coord = y_coords[y == cls]
		color = colors[idx]
		plt.scatter(x_coord, y_coord, color = color)

	plt.xlabel('Component I')
	plt.ylabel('Component II')

	if title is not None:
		plt.title(title)

	plt.show()

def iris_classification():
	print('\nIris classification using GMM\n')
	print('Initiating Data Load...')
	iris = datasets.load_iris()
	X, y = iris.data, iris.target

	# X, y = datasets.make_blobs()

	size = len(X)
	indices = list(range(size))
	np.random.shuffle(indices)
	X, y = np.array([X[idx] for idx in indices]), np.array([y[idx] for idx in indices])

	train_size = int(0.8 * len(X))
	X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

	print('Data load complete!')

	num_classes = max(y) + 1

	print('Constructing Gaussian Mixture Model...')
	classifier = GMM(num_states = num_classes)
	classifier.fit(X_train)

	print('Generating test predictions...')
	predictions = classifier.predict(X_test)

	print(predictions, y_test)
	
	plot_results(X_test, y_test, title = 'Input Clusters')
	plot_results(X_test, predictions, title = 'GMM Clusters')

def main():
	np.random.seed(3)
	iris_classification()


if __name__ == '__main__':
	main()
