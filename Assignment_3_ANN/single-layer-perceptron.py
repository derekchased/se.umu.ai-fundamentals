
import numpy as np
import helper_functions as hf

# Step 1, Load and prep the Data (shuffle, normalize, split)
X_train, Y_train, X_test, Y_test = hf.load_and_prep_test_train_data()

# Step 2, One Hot Encode Y labels- Convert each class into it's own binary feature
Y_labels_one_hot = hf.one_hot_encode_labels(Y_train)

print("Y_labels_one_hot\n",Y_labels_one_hot)


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
print("weights", weights.shape,"\n",weights)
print("\nweights.T", weights.T.shape,"\n",weights.T)

# Step 4, Choose learning rate
learning_rate = .01


# Step 5, 
#	Outer loop- Iterate over the training data MAX_ITER num times
#	Inner loop- Iterate over each sample in the data, update weights
#		at each iteration
MAX_ITER = 1000
MAX_ITER = 1

#NUM_SAMPLES = X_train.shape[0]
NUM_SAMPLES = 5

for m in range(MAX_ITER):

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
		activation = 1/(1+np.exp(-weights_transpose_dot_x_i))
		
		# Step 4, Calculate the error

		#	a) Get the difference between the correct label
		#	   and the activation function
		error = Y_labels_one_hot[i] - activation

		# 	b) Multiply by the learning rate
		error2 = learning_rate*error

		# 	c) Multiply by the activation
		error3 = error2*activation

		#	d) Multiply by the current weights
		error4 = error3*weights_transpose_dot_x_i

		# Step 5, Add the current weights with the error
		#	Do error4.reshape(-1,1) so numpy can broadcast 
		#	the operation to all elements in the weight vector
		#	error4.reshape(-1,1).shape = (4,1)
		#	weights.T.shape = (4,784)
		#	... so it will add error4[0] to all weights.T[0,:]
		#	... so it will add error4[1] to all weights.T[1,:]
		new_weights = weights.T + error4.reshape(-1,1)

		# Step 6, Assign the new weights to the weights
		weights = new_weights.T
		
		print("\n")
		print(i,"update weights")
		print("current_x", current_x.shape)
		print("Y_labels_one_hot[",i,"]", Y_labels_one_hot[i])
		"""print("current_x.shape",current_x.shape)
								print("weights_transpose_dot_x_i.shape",weights_transpose_dot_x_i.shape)"""
		print("activation", activation)
		print("error", error)
		print("error2", error2.shape, error2)
		"""print("error3", error3)
		print("error4", error4.shape, error4)
		print("error4.reshape(-1,1)",error4.reshape(-1,1).shape, error4.reshape(-1,1))
		print("weights = new_weights.T",new_weights.T.shape)"""

# Step 7, Get Final activations for each sample accuracy
activations = []

#	Iterate over each sample
for i in range(X_train.shape[0]):
	
	# Get dot product of weights and the sample's features
	weights_transpose_dot_x_i = np.dot(weights.T,X_train[i,:]);
	
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
label_indexes = np.argwhere(Y_labels_one_hot==1 )[:,1]

#	Compare Y activated to actual YGet index of correct label
prediction_correct = activated_indexes==label_indexes

#	Count number of correct activations
num_correct = np.sum(prediction_correct == True)

#	Calculate the accuracy of the classifier
accuracy = num_correct/len(prediction_correct)
