import numpy as np
import sys
import matplotlib.pyplot as plt
import helper_functions as hf

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

def predict(X, W):
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

	# 	Convert index to digit predictions
	digit_predictions = np.copy(activated_indexes)
	digit_predictions[ digit_predictions == 0 ] = 4
	digit_predictions[ digit_predictions == 1 ] = 7
	digit_predictions[ digit_predictions == 2 ] = 8
	digit_predictions[ digit_predictions == 3 ] = 9

	return activated_indexes, digit_predictions



def run_main(training_images_file_path, training_labels_file_path, validation_images_file_path, validation_labels_file_path):
	# Step 1, Load and prep the Data (shuffle, normalize, split)

	LABRES = (validation_labels_file_path == "")
	# LABRES = True

	if(LABRES):
		#	Live, use all the data for training
		X_train, Y_train, X_test, Y_test = hf.load_and_prep_test_train_data(training_images_file_path, training_labels_file_path, 1)
		
		# 	Load validation data
		X_val = hf.load_validation_data(validation_images_file_path)

	else:
		#	Testing, use .8 for training data
		X_train, Y_train, X_test, Y_test = hf.load_and_prep_test_train_data(training_images_file_path, training_labels_file_path,.8)
		
		# 	Load validation data and labels
		X_val, Y_val = hf.load_and_test_validation_data(validation_images_file_path, validation_labels_file_path)

		# 	Encode labels to binary features
		Y_labels_test_one_hot = hf.one_hot_encode_labels(Y_test)

		# 	Load validation data and labels
		Y_labels_validation_one_hot = hf.one_hot_encode_labels(Y_val)


	# Step 2, One Hot Encode Y labels- Convert each class into it's own binary feature
	Y_labels_one_hot = hf.one_hot_encode_labels(Y_train)
	

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
	weights = hf.get_random_weights(weights_matrix_dimension)
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
	EPSILON = .01

	# Keep track of error/loss
	iteration_errors = []
	train_accuracy = []

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

			# 	b) Multiply by the learning rate
			error2 = learning_rate*error

			# 	c) Multiply by the activation
			error3 = error2.reshape(-1,1)*current_x

			#	d) Add this sample's error to aggregate error for this iteration
			curr_iteration_aggregate_error = curr_iteration_aggregate_error + np.sum(error3)


			# Step 5, Add the current weights with the error
			new_weights = weights.T + error3

			# Step 6, Assign the new weights to the weights
			weights = new_weights.T
			

		# THIS IS A VERY BASIC EARLY STOPPING IMPLEMENTATION
		
		# Keep track of aggregate error
		iteration_errors.append(curr_iteration_aggregate_error)
		train_accuracy.append(get_accuracy(X_train,weights, Y_labels_one_hot))

		# Do some minimal training before considering early stopping
		if len(iteration_errors) > MIN_ITER:

			# Get the difference between current and previous epoch errors
			diff = abs( iteration_errors[-1] - iteration_errors[-2]  )
			
			# The diff is 'how much did the loss improve'
			# So we check if the improvement is less than the minimum improvement value epsilon
			# If it is less, then we break out of training
			if(diff<EPSILON):
				print("break",m)
				print("diff",diff)
				print("EPSILON",EPSILON)
				print("MIN_ITER",MIN_ITER)
				print("MAX_ITER",MAX_ITER)
				break;
	
	# This is a check to see if running live on Labres or at home
	if(LABRES):
		indexes, predictions = predict(X_val, weights)
		print(*predictions, sep = "\n")

	else:
		print("==RESULTS==")
		print("train epochs", m)
		print("train accuracy",get_accuracy( X_train, weights, Y_labels_one_hot))
		print("test accuracy",get_accuracy(X_test, weights, Y_labels_test_one_hot))
		print("val accuracy",get_accuracy(X_val, weights, Y_labels_validation_one_hot))

		iteration_errors.append(curr_iteration_aggregate_error)
		train_accuracy.append(get_accuracy(X_train,weights, Y_labels_one_hot))

		plt.figure(figsize=(1,1),num=1) # num markers/ticks/steps on the graph [0,1]

		print( "derek", len(iteration_errors), len(X_train))
		print("iteration_errors", iteration_errors)
		print("X_train",X_train) 

		plt.plot(np.arange(len(iteration_errors)),np.abs(iteration_errors)/len(X_train),c='g') 
		plt.title("Decreasing Train Error")
		plt.xlabel('Iterations')
		plt.ylabel('Error')
		plt.show()
		plt.title("Increasing Train Accuracy")
		plt.plot(np.arange(len(train_accuracy)),train_accuracy,c='g') 
		plt.xlabel('Iterations')
		plt.ylabel('Accuracy')
		plt.show()
	
if __name__ == "__main__":
	if(len(sys.argv)>1):
		training_images_file_path = sys.argv[1]
		training_labels_file_path = sys.argv[2]
		validation_images_file_path = sys.argv[3]
		validation_labels_file_path = ""
	else:
		training_images_file_path = "./MNIST/training-images.txt"
		training_labels_file_path = "./MNIST/training-labels.txt"
		validation_images_file_path = "./MNIST/validation-images.txt"
		validation_labels_file_path = "./MNIST/validation-labels.txt"

	run_main(training_images_file_path, training_labels_file_path, validation_images_file_path, validation_labels_file_path)