import numpy as np
import sys

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
	# randominze between -.5, .5
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

def get_accuracy(X, W, Y):
	# Step 7, Get Final activations for each sample accuracy
	activations = []

	#	Iterate over each sample
	for i in range(X.shape[0]):
		
		# Get dot product of weights and the sample's features
		weights_transpose_dot_x_i = np.dot(W.T,X[i,:]);
		
		# Take the activation function
		activation = 1/(1+np.exp(-weights_transpose_dot_x_i))

		# Append the activation function of this sample to the list
		activations.append(activation)

	# Step 8, Analyze activations

	#	Convert list to vector
	activations_vector = np.asarray(activations);

	#	Get index of which node was activated for each sample
	activated_indexes = np.argmax(activations_vector,1)

	#	Get index of correct label
	label_indexes = np.argwhere(Y ==1 )[:,1]

	#	Compare Y activated to actual YGet index of correct label
	prediction_correct = activated_indexes==label_indexes

	#	Count number of correct activations
	num_correct = np.sum(prediction_correct == True)

	# Total samples
	total_samples = len(prediction_correct)
	
	# Accuracy
	accuracy = num_correct/total_samples

	return accuracy



def run_main():
	# Step 1, Load and prep the Data (shuffle, normalize, split)
	X_train, Y_train, X_test, Y_test = load_and_prep_test_train_data()

	# Step 2, One Hot Encode Y labels- Convert each class into it's own binary feature
	Y_labels_one_hot = one_hot_encode_labels(Y_train)
	Y_labels_test_one_hot = one_hot_encode_labels(Y_test)

	#print("Y_labels_one_hot\n",Y_labels_one_hot)


	# Step 3, Create weights matrix
	# the weights matrix is [ 784 x 4]
	# therefore the matrix is arranged by feature as rows and output nodes in the columns
	# ... so [0,0] is the weight from feature 0 to node 0, [0,1] feature 0 to node 1, ... [0,3] 
	# from feature 0 to the final node 3
	# when we calculate further below we will use the transpose of this matrix as we are interested in
	# the weights coming into the node [to output node, from input node]

	# 	a) Setup dimensions (num features in each sample X number classes or output nodes)
	weights_matrix_dimension = (X_train.shape[1], len(Y_labels_one_hot[0]))

	#	b) Fill with random values
	weights = get_random_weights(weights_matrix_dimension)
	#print("weights", weights.shape,"\n",weights)
	#print("\nweights.T", weights.T.shape,"\n",weights.T)

	# Step 4, Choose learning rate
	learning_rate = .01


	# Step 5, 
	#	Outer loop- Iterate over the training data MAX_ITER num times
	#	Inner loop- Iterate over each sample in the data, update weights
	#		at each iteration
	#MAX_ITER = 1000
	MAX_ITER = 1000
	MIN_ITER = 5

	NUM_SAMPLES = X_train.shape[0]

	# Early  stopping
	EPSILON = .001

	# Keep track of error/loss
	iteration_errors = []

	# Keep track of epoch for early stopping
	m = 0

	for m in range(MAX_ITER):
		
		# 	Create local error list for this iteration, used to aggregate
		curr_iteration_aggregate_error = 0

		# Step 6, Iterate over each sample in X
		for i in range(NUM_SAMPLES):
			
			# Forward Algorithm
			# 	using matrix math to calculate the weights, activation, 
			# 	and error for all output nodes at once

			# Step 1, get the current X sample
			current_x = X_train[i,:]
			
			# Step 2, Get the dot product of the transpose of
			# the weights matrix and the current X sample
			weights_transpose_dot_x_i = np.dot(weights.T, current_x);

			# Step 3, Get the activation of each output node
			activation = 1/(1+np.exp( -weights_transpose_dot_x_i))
			
			# Step 4, Calculate the error

			#	a) Get the difference between the correct label
			#	   and the activation function
			error = Y_labels_one_hot[i] - activation  
			#labels [4]

			#			[1    0     0    0]
			#activation [1  .09   .73   .01    ]
				#		 0  -.09  -.73   -.01
			# 	b) Multiply by the learning rate
			error2 = learning_rate*error

			# 	c) Multiply by the activation
			error3 = error2.reshape(-1,1)*current_x#error2*activation

			#	d) Add this sample's error to aggregate error for this iteration
			curr_iteration_aggregate_error = curr_iteration_aggregate_error + np.sum(error3)


			#	d) Multiply by the current weights
			#error4 = error3*weights_transpose_dot_x_i

			# Step 5, Add the current weights with the error
			#	Do error4.reshape(-1,1) so numpy can broadcast 
			#	the operation to all elements in the weight vector
			#	error4.reshape(-1,1).shape = (4,1)
			#	weights.T.shape = (4,784)
			#	... so it will add error4[0] to all weights.T[0,:]
			#	... so it will add error4[1] to all weights.T[1,:]
			#new_weights = weights.T + error4.reshape(-1,1)
			new_weights = weights.T + error3#.reshape(-1,1)

			# Step 6, Assign the new weights to the weights
			weights = new_weights.T
			
			#print("weights_transpose_dot_x_i.shape",weights_transpose_dot_x_i.shape)
			#print("activation", activation)
			#print("error", error)

			"""print("\n")
			print(i,"update weights")
			print("current_x", current_x.shape)
			print("Y_labels_one_hot[",i,"]", Y_labels_one_hot[i])
			print("current_x.shape",current_x.shape)
			print("weights_transpose_dot_x_i.shape",weights_transpose_dot_x_i.shape)
			print("activation", activation)
			print("error", error)
			print("error>=0", error>=0)
			print("error2", error2)
			#print("\nerror2", error2.shape,error2)
			#print("aaa", (error2.reshape(-1,1)*current_x).shape)
			#print("error2", error2.reshape(-1,1))
			#print("error2*weights.T", error2[0]*weights.T[0])
			
			print("i")
			print("sum curr errors", np.sum(error3))
			print("error aggregate", np.sum(curr_iteration_aggregate_error))
			#print("error3", error3.shape,error3)
			#print("error3[0]", error3[0])
			#print("error4", error4.shape, error4)
			#print("error4.reshape(-1,1)",error4.reshape(-1,1).shape, error4.reshape(-1,1))
			#print("weights = new_weights.T",new_weights.T.shape)"""


		# THIS IS A VERY BASIC EARLY STOPPING IMPLEMENTATION
		
		# Keep track of aggregate error
		iteration_errors.append(curr_iteration_aggregate_error)

		# Do some minimal training before considering early stopping
		if len(iteration_errors) > MIN_ITER:

			# Get the difference between current and previous epoch errors
			diff = abs( iteration_errors[-1] - iteration_errors[-2]  )
			# print("diff",diff)
			# print("<eps",diff<EPSILON)
			# print("train accuracy",get_accuracy(X_train,weights, Y_labels_one_hot))
			# print("test accuracy ",get_accuracy(X_test,weights,  Y_labels_test_one_hot))

			# The diff is 'how much did the loss improve'
			# So we check if the improvement is less than the minimum improvement value epsilon
			# If it is less, then we break out of training
			if(diff<EPSILON):
				#print("break",m)
				break;
		

	#print(iteration_errors)
	#diffs = np.diff(iteration_errors)
	#print("diffs")
	#print(diffs[-100:])
	#print(iteration_errors[-1])
	#print(iteration_errors[-2])
	#print(np.diff(iteration_errors[-1],iteration_errors[-2]))

	print("==RESULTS==")
	print(m)
	print("train accuracy",get_accuracy( X_train, weights, Y_labels_one_hot))
	print("test accuracy",get_accuracy(X_test, weights, Y_labels_test_one_hot))


if __name__ == "__main__":
	print('Number of arguments:', len(sys.argv), 'arguments.')
	print('Argument List:', str(sys.argv))
	run_main()