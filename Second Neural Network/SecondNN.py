import numpy as np 
import random
import matplotlib.pyplot as plt
import math

# 5 input neurons in the input layer
# 2 hidden layers with 6 neurons each
# 2 output neurons in the output layer
# all activation functions are sigmoid


dataset_unfiltered = np.array([
	[0,		0,		0,		0,		0,		0,		1],
	[0,		0,		0,		0,		1,		0,		1],
	[0,		0,		0,		1,		0,		0,		1],
	[0,		0,		0,		1,		1,		0,		1],
	[0,		0,		1,		0,		0,		0,		1],
	[0,		0,		1,		0,		1,		0,		1],
	[0,		0,		1,		1,		0,		0,		1],
	[0,		0,		1,		1,		1,		0,		1],
	[0,		1,		0,		0,		0,		1,		0],
	[0,		1,		0,		0,		1,		0,		1],
	[0,		1,		0,		1,		0,		1,		0],
	[0,		1,		0,		1,		1,		1,		0],
	[0,		1,		1,		0,		0,		0,		1],
	[0,		1,		1,		0,		1,		0,		1],
	[0,		1,		1,		1,		0,		1,		0],
	[0,		1,		1,		1,		1,		1,		0],
	[1,		0,		0,		0,		0,		0,		1],
	[1,		0,		0,		0,		1,		0,		1],
	[1,		0,		0,		1,		0,		1,		0],
	[1,		0,		0,		1,		1,		0,		1],
	[1,		0,		1,		0,		0,		1,		0],
	[1,		0,		1,		0,		1,		1,		0],
	[1,		0,		1,		1,		0,		1,		0],
	[1,		0,		1,		1,		1,		1,		0],
	[1,		1,		0,		0,		0,		0,		1],
	[1,		1,		0,		0,		1,		0,		1],
	[1,		1,		0,		1,		0,		1,		0],
	[1,		1,		0,		1,		1,		0,		1],
	[1,		1,		1,		0,		0,		1,		0],
	[1,		1,		1,		0,		1,		1,		0],
	[1,		1,		1,		1,		0,		1,		0],
	[1,		1,		1,		1,		1,		1,		0]
	])


def generateRandom2DMatrix(m, n):
	return [ [random.randint(1,100)/100 for k in range(n)] for i in range(m) ]

def sigmoid(z):
	return 1/(1+np.exp(-z))

def CostFunction(NN_output,output):
	return output*(np.log10(NN_output)) + (1-output)*(np.log10(1 - NN_output))


data_set = [ [0 for k in range(6)] for i in range(len(dataset_unfiltered)) ]
index_data_set = 0



for k in range(len(dataset_unfiltered[:,0])):

	for i in range(dataset_unfiltered[k][5]):
		#temp = np.append( dataset_unfiltered[k][0:5], 1)
		data_set[index_data_set] = np.append( dataset_unfiltered[k][0:5], 1)
		index_data_set += 1

	for j in range(dataset_unfiltered[k][6]):
		#temp = dataset_unfiltered[k][0:5]
		data_set[index_data_set] = np.append( dataset_unfiltered[k][0:5], 0)
		index_data_set += 1


#for i in range(1700,1799):
#	print("#",i,"  ", data_set[i])


# Splitting dataset into training dataset and testing dataset:
# But first, we need to shuffle the entire dataset!

random.seed(99)
random.shuffle(data_set)

training_data_set = data_set[0:(math.ceil(0.70*len(data_set)))]
testing_data_set  =	data_set[(math.ceil(0.70*len(data_set))): len(data_set)]



#for i in range(1700,1799):
#	print("#",i,"  ", training_data_set[i])

print("Training dataset:   ", len(training_data_set))
print("Testing dataset:    ", len(testing_data_set))

inputs = [ [0 for k in range(5)] for i in range(len(training_data_set)) ]
desired_outputs = [0 for k in range(len(training_data_set))]


inputs_testing = [ [0 for k in range(5)] for i in range(len(testing_data_set)) ]
desired_outputs_testing = [0 for k in range(len(testing_data_set))]



for i in range(len(training_data_set)):
	inputs[i] = training_data_set[i][0:5]
	desired_outputs[i] = training_data_set[i][5]

for i in range(len(testing_data_set)):
	inputs_testing[i] = training_data_set[i][0:5]
	desired_outputs_testing[i] = training_data_set[i][5]

bias_space = 1

nInputs_neurons 		= 5	+ bias_space
nNeurons_second_layer 	= 6	+ bias_space
nNeurons_third_layer  	= 6	+ bias_space
output_neurons 			= 1	+ bias_space

Weights = [ [] for i in range(3)]

#the first column is the Bias. So it will be set to 1.
Weights[0] = np.array(generateRandom2DMatrix(nNeurons_second_layer, nInputs_neurons ))
Weights[1] = np.array(generateRandom2DMatrix(nNeurons_third_layer, nNeurons_second_layer ))
Weights[2] = np.array(generateRandom2DMatrix(output_neurons, nNeurons_third_layer ))

Weights = np.array(Weights)

#Weights[0][0,:] = 0
#Weights[1][0,:] = 0
#Weights[2][0,:] = 0


input_layer 		= [ 0 for i in range(nInputs_neurons)]
first_hidden_layer 	= [ 0 for i in range(nNeurons_second_layer)]
second_hidden_layer = [	0 for i in range(nNeurons_third_layer)]
output_layer 		= [	0 for i in range(output_neurons)]



# These are bias
input_layer = np.append(1, inputs[0])
first_hidden_layer[0] = 1
second_hidden_layer[0] = 1

neurons = [input_layer, first_hidden_layer, second_hidden_layer, output_layer]
neurons = (neurons)
#neuron["layer"]["Which neuron of the layer/ FIRST ONE IS THE BIAS"]
neurons[0][0] = 1
neurons[1][0] = 1
neurons[2][0] = 1
neurons[3][0] = 1
#print(Weights)


def forwardPropagation(data_number = 0, training_data = True, showComputation = False, matrix_neurons = neurons, W = Weights):
	
	if(training_data == True):
		input_i = inputs[data_number]
	else:
		input_i = inputs_testing[data_number]

	matrix_neurons[0] = np.append(1 , input_i)
	matrix_neurons[1][0] = 1
	matrix_neurons[2][0] = 1
	matrix_neurons[3][0] = 1
	
	if(showComputation == True):
		print(matrix_neurons)

	for layer in range(len(W)):
		for neuron_j in range(len(matrix_neurons[layer+1])):
			next_layer_neuron = layer + 1
			z_j = np.dot( matrix_neurons[layer] , W[layer][neuron_j,:] )
			matrix_neurons[next_layer_neuron][neuron_j] = sigmoid(z_j)
			if(showComputation == True):
				print(matrix_neurons[layer] ," . " , W[layer][neuron_j,:], "  = s([",next_layer_neuron, "][",neuron_j,"]]) = ", sigmoid(z_j))
				print("")

	matrix_neurons[1][0] = 1
	matrix_neurons[2][0] = 1
	matrix_neurons[3][0] = 1

	return matrix_neurons, W

	

# test
forwardPropagation(0)


def printNeurons(matrix_neurons = neurons):
	layer_names = ["Inputs        ", "Hidden layer 1", "Hidden layer 2", "Outputs       "]
	for i in range(len(matrix_neurons)):
		print(layer_names[i] , ":        ", matrix_neurons[i])


# Initializing delta
def get_copy_of_neurons_matrix_null(matrix):
	m = matrix.copy()
	for i in range(len(m)):
		m[i] = np.matrix(m[i])
		m[i].fill(0)
	return m



delta_error_matrix = get_copy_of_neurons_matrix_null(neurons)



def computeDeltaMatrix(training_example = 0, showComputation = False, W = Weights, matrix_neurons = neurons, Delt = delta_error_matrix):
	
	if(showComputation == True):
		print("STARTING: ")
	#training_example = 1000
	input_i = inputs[training_example] 
	desired_outputs_i = desired_outputs[training_example]

	if(showComputation == True):
		print(W)
		print("Training example: ", training_example)
		print("Inputs and outputs: ", input_i, desired_outputs_i)

	for layer in reversed(range(1,len(matrix_neurons))):
		if(showComputation == True):
			print(layer)
		if(layer == len(matrix_neurons) - 1):
			if(showComputation == True):
				print(np.subtract(matrix_neurons[layer][1:len(matrix_neurons[layer])] , desired_outputs_i))
			Delt[layer] = np.subtract(matrix_neurons[layer][1:len(matrix_neurons[layer])] , desired_outputs_i)
			Delt[layer] = np.append(0, Delt[layer])
		else:
				w = W[layer].copy() #[1:len(W[layer])].copy()
				
				w_T = w.transpose() 
				if(showComputation == True):
					print("w: ", w)
					print("w_T:")
					print(w_T)
					print("delta ",layer + 1,":" )
					print(Delt[layer + 1])
				x = np.matmul(w_T, Delt[layer + 1])
				#x = x[1:len(x)]

				# Derivative_Sigmoid_of_neuron_layer(z) = neurons_of_layer*(1-neurons_of_layer)
				neurons_of_layer = np.array(matrix_neurons[layer].copy())#[1:len(matrix_neurons[layer])].copy())
				vector_of_one 	 = np.array(neurons_of_layer.copy())
				vector_of_one.fill(1)
				y = np.subtract(vector_of_one, neurons_of_layer)
				Derivative_Sigmoid_of_neuron_layer = np.multiply(neurons_of_layer , y)

				# Putting the whole equation together:
				Delt[layer] = np.multiply(x , Derivative_Sigmoid_of_neuron_layer)
		if(showComputation == True):
			print("delta[",layer,"]", Delt[layer])
	return Delt




delta = computeDeltaMatrix() 
#print(np.matrix(delta))


def training(nomber_of_training_set = len(training_data_set), matrix_neurons = neurons, W = Weights, regularization_param = 5):
	mat_weights_null = get_copy_of_neurons_matrix_null(W)
	
	for training_set in range(nomber_of_training_set):
		#print(training_set)
		matrix_neurons, W 	= forwardPropagation(training_set, True, False, matrix_neurons, W)
		delta 				= computeDeltaMatrix(training_set)

		# Backpropagation algorithm
		for layer in reversed(range(1, len(matrix_neurons))):
			neuron_vector = neurons[layer-1].copy()
			delta_vector  = delta[layer].copy()

			neuron_vector = np.matrix(neuron_vector)
			delta_vector  = np.matrix(delta_vector).transpose()

			multiplication_term = np.matmul(delta_vector, neuron_vector)
			mat_weights_null[layer-1] = mat_weights_null[layer-1] + multiplication_term

	c = (1/nomber_of_training_set)
	#derivative_J = (1/nomber_of_training_set)*mat_weights_null
	derivatives_Cost_Weight = get_copy_of_neurons_matrix_null(W)

	for layer in range(len(mat_weights_null)):
		for m in range(len(mat_weights_null[layer][:,0])):
			for n in range(len(mat_weights_null[layer][0,:])):
				if(n != 0):
					derivatives_Cost_Weight[layer][m][n] = c*mat_weights_null[layer][m][n] + regularization_param*W[layer][m][n]
				else:
					derivatives_Cost_Weight[layer][m][n] = c*mat_weights_null[layer][m][n]
	return derivatives_Cost_Weight


derivatives_Cost_Weight = training()

print(Weights)


def gradient_descend(iterations = 10000, alpha = 0.01, dW_matrix = derivatives_Cost_Weight):
	dW_matrix = np.array(dW_matrix)
	for i in range(len(derivatives_Cost_Weight)):
		dW_matrix[i] = np.array(dW_matrix[i])
	#m , w =  forwardPropagation()
	#current_cost = CostFunction(neurons[3][1], desired_outputs[0])
	#prev_cost	 = current_cost+1



	i = 0
	j = 0
	#while((prev_cost >= current_cost) or j <= 10000):
		#prev_cost = current_cost
	for i in range(5000):
		for layer in range(len(dW_matrix)):
			for m in range(len(dW_matrix[layer][:,0])):
				for n in range(len(dW_matrix[layer][0,:])):
					a = dW_matrix[layer][m][n]
					j = Weights[layer][m][n]
					Weights[layer][m][n] = j - alpha*a
		#m, w = forwardPropagation(i%len(testing_data_set), True, False, m, w)
		#current_cost = CostFunction(neurons[3][1], desired_outputs[i%len(testing_data_set)])
		#i += 1
		#j += 1


gradient_descend()
print(Weights)

NN_all_training_outputs = [ [] for x in range(len(training_data_set))]
print(len(NN_all_training_outputs))
print((NN_all_training_outputs))
mat, w = forwardPropagation(i, True, False, neurons, Weights)

#NN_outputs = [ [] for i in range(len(training_data_set))]
#NN_outputs = np.array(NN_outputs)

for i in range(len(training_data_set)):
	#print(inputs[i])
	mat, w = forwardPropagation(i, True, False, mat, Weights)
	#NN_outputs[i][0] = desired_outputs[i]
	#NN_outputs[i][1] = mat[3][1]
	#print("Network output: ", mat[3][1], "   Desired output: ", desired_outputs[i], "   ", CostFunction(mat[3][1], desired_outputs[i]))

#print(NN_outputs)
















