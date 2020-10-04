
# linear algebra package for Python
import numpy as np

# Plotting and analysis package for python
# import matplotlib.pyplot as plt

def load_and_prep_test_train_data():
	
	# Load data from txt files
	X, Y = load_and_prep_data("./MNIST/training-images.txt", "./MNIST/training-labels.txt")

	# Split into test and train
	X_train, Y_train, X_test, Y_test = train_test_split(X, Y);

	return X_train, Y_train, X_test, Y_test


def load_and_prep_validation_data():
	
	# Load data from txt files
	X, Y = load_and_prep_data("./MNIST/validation-images.txt", "./MNIST/validation-labels.txt")

	return X, Y


def load_and_prep_data(X_filename, Y_filename):
	# Load the data, skip first 3 rows which are comments
	X = np.loadtxt(X_filename, skiprows=3) # load csv
	Y = np.loadtxt(Y_filename, skiprows=3) # load csv

	# Randomize the data
	X, Y = randomize_data(X, Y);

	# Normalize the data
	X = normalize_data(X)

	return X, Y
	
def normalize_data(X):
	X_normalized = X/255
	return X_normalized

def randomize_data(X, y, seed=7):
    # Create generator object with seed (for consistent testing across compilation)
    #gnrtr = np.random.default_rng(7)
    np.random.seed(seed)

    # Create random array with values permuted from the num elements of y
    #r = gnrtr.permutation(len(y))
    r = np.random.permutation(len(y))

    # Reorganize X and y based on the random permutation, all columns
    return X[r, :], y[r]

def train_test_split(X, Y, train_percent=.8):
	train_size = int(len(X)*train_percent)
	X_train = X[:train_size, :]
	X_test = X[train_size:, :]
	Y_train = Y[:train_size]
	Y_test = Y[train_size:]
	return X_train, Y_train, X_test, Y_test

def get_random_weights(weight_dimensions):
	return np.random.random_sample(weight_dimensions)

def one_hot_encode_labels(Y):
	#	a) Get unique Y Labels
	Y_labels = np.unique(Y)

	#	b) Create zero array with dimension (length of train samples, num unique labels)
	Y_labels_one_hot = np.zeros(  (  len(Y), len(Y_labels) ))

	#	c) Mark the correct feature with a 1
	for index, classification in enumerate(Y_labels):
		#print("index",index, "classification", classification )
		Y_labels_one_hot[np.argwhere(Y == classification),index] = 1

	return Y_labels_one_hot