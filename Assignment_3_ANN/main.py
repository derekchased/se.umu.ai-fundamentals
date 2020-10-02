
# linear algebra package for Python
import numpy as np

# Plotting and analysis package for python
# import matplotlib.pyplot as plt

def load_data():
	# Load the data, skip first 3 rows which are comments

	X_train = np.loadtxt("./MNIST/training-images.txt", skiprows=3) # load csv
	Y_train = np.loadtxt("./MNIST/training-labels.txt", skiprows=3) # load csv

	X_test = np.loadtxt("./MNIST/validation-images.txt", skiprows=3) # load csv
	Y_test = np.loadtxt("./MNIST/validation-labels.txt", skiprows=3) # load csv

	# check the dimensions
	# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

	return X_train, Y_train, X_test, Y_test

def fit(classifier, X, Y):
	# take data X with labels Y and build classifier
	return classifier

def predict(classifier, X):
	# use a fitted classifier to predict, Z, the labels of X
	return Z

def score_accuracy(Y, Z):
	# compare the actual labels Y with the predicted labels Z and return the score
	# assume len(Y) == len(Z)

	# sum of the correct predictions divided by total rows
	return np.sum( Y == Z ) / len(Y)


# Step 1, Load the Data
X_train, Y_train, X_test, Y_test = load_data()

# Step 2, Create ANN Architecture
# classifier = ANN(num_input_layers, num_hidden_layers, num_output_layers, weights, biases)

# Step 3, Fit the data
# classifier.fit(X_train, Y_train)

# Step 3, Predict the test (validation) data
# Z_predicted = classifier.predict(X_test)

# Step 4, Get accuracy 
# classifer.score(Y_test, Z_predicted)

