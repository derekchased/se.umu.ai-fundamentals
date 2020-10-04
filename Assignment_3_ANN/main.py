
# linear algebra package for Python
import numpy as np

# Plotting and analysis package for python
# import matplotlib.pyplot as plt

def load_data():
	# Load the data, skip first 3 rows which are comments
	X = np.loadtxt("./MNIST/training-images.txt", skiprows=3) # load csv
	Y = np.loadtxt("./MNIST/training-labels.txt", skiprows=3) # load csv

	# Randomize the data
	X, Y = randomize_data(X, Y);

	# Split into test and train
	X_train, Y_train, X_test, Y_test = train_test_split(X, Y);

	# Load validation data
	X_validate = np.loadtxt("./MNIST/validation-images.txt", skiprows=3) # load csv
	Y_validate = np.loadtxt("./MNIST/validation-labels.txt", skiprows=3) # load csv

	# check the dimensions
	print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, X_validate.shape, Y_validate.shape)

	return X_train, Y_train, X_test, Y_test, X_validate, Y_validate

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

"""def fit(classifier, X, Y):
	# take data X with labels Y and build classifier
	return classifier

def predict(classifier, X):
	# use a fitted classifier to predict, Z, the labels of X
	return Z

def score_accuracy(Y, Z):
	# compare the actual labels Y with the predicted labels Z and return the score
	# assume len(Y) == len(Z)

	# sum of the correct predictions divided by total rows
	return np.sum( Y == Z ) / len(Y)"""



def ann(layer_sizes, max_iterations, X, Y):
	print("ann")
	print(layer_sizes)

	# Create list of all layers. X is the first layer
	layers = [X]
	weights = []


def forward(X, A):
	current_layer = X
	next_layer = A
	weight_dimensions = (len(next_layer), len(current_layer))
	weights = np.random.random_sample(weight_dimensions)
	
	

	"""
			
				for h in layer_sizes:
								
								# init each layer vector with empty values, with num nodes = h
								layers.append( np.empty(h) )
			
				for i in range(len(layers)-1):
					# init each weight-layer with dimensions (num nodes going to, num input)
					weights.append(np.random.random_sample(  ( len(layers[i+1]), len(layers[i]) ) ))
			
				for l in layers:
					print(l.shape, l)
			
				for w in weights:
					print(w.shape, w)
				# Create Weights
			
				return []"""


# Step 1, Load the Data
X_train, Y_train, X_test, Y_test, X_validate, Y_validate = load_data()

# Step 2, Create ANN Architecture
#print(Y_train)
ann( (X_train.shape[1],  10, 10, len( np.unique(Y_train) ) ), 100, X_train, Y_train)
# classifier = ANN(num_input_layers, num_hidden_layers, num_output_layers, weights, biases)

# Step 3, Fit the data
# classifier.fit(X_train, Y_train)

# Step 3, Predict the test (validation) data
# Z_predicted = classifier.predict(X_test)

# Step 4, Get accuracy 
# classifer.score(Y_test, Z_predicted)

