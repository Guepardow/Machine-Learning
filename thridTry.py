import numpy as np
import random

# Activation function will be sigmoid
def sigmoid(x, derivative = False):
	if(derivative == True):
		s = sigmoid(x,False)
		return s*(1 - s)
	return 1/(1+np.exp(-x))

def deriveSigmoid(x):
	return sigmoid(x)*(1 - sigmoid(x))

def Cost(last_layer_neurons_vector, desired_output_vector):
	cost = 0
	for i in range(len(last_layer_neurons_vector)):
		#print('(', last_layer_neurons_vector[i], '-', desired_output_vector[i], ")^2   +", end='')
		cost += (last_layer_neurons_vector[i] - desired_output_vector[i])**2
	return cost

def local_cost(nn_output, desired_output):
	return (nn_output - desired_output)**2

def generate2DMatrix(m, n, num = 0):
	return [ [random.randint(0,100)/100 for k in range(n)] for i in range(m) ]

def generate3DMatrix(m,n,k):
	return [[[random.randint(0,100)/100 for k in range(n)] for j in range(m)] for i in range(k)]

# Get random Weights and Bias for the neural network
# inputs = 4
# nomber of hidden layers = 2
# nomber of neurons in each hidden layer = 4

# How to make a 2D array/list/matrix (m x n):
# hidden_neurons = [[0]*nomber_of_neurons_in_one_hidden_layer for i in range(nomber_of_hidden_layers)]

#matrix_neurons = np.array(generate2DMatrix(3,4))
#print("Initializing neuron matrix...")
#print(matrix_neurons)
#print("")


def generateRandomWeights(mat_neurons):
	Layers = len(mat_neurons[0])
	synapses_layers = Layers - 1
	Weights = generate3DMatrix(len(mat_neurons[:,0]), len(mat_neurons[:,0]), synapses_layers)

	#Weights.append(generate2DMatrix(len(mat_neurons[i+1,:]), len(mat_neurons[i,:]), 1))
	return Layers, synapses_layers, Weights


#layers, syn_layers, Weights = generateRandomWeights(matrix_neurons)

Weights = np.array(
				  [[[0.16, 0.71, 0.24],
				  [0.48, 0.26, 0.83],
				  [0.15 ,0.66, 0.23]
				  ]
				  ,
				 [[0.41, 0.4 , 0.64],
				  [0.65, 0.05 ,0.69],
				  [0.26 ,0.26 ,0.17]]
				  ,
				 [[0.37, 0.06, 0.52],
				  [0.18, 0.84, 0.94],
				  [0.42, 0.44, 0.24]]])

matrix_neurons = np.array([[1,   0.09, 0.5,  0.1 ],
						 [0,   0.18, 0.9,  0.38],
						 [1,   0.38, 0.42, 0.54]])

layers = 4

print("")
print("Initialializing random weights...")
Weights = np.array(Weights)
print("")
print(Weights)
print("")
print("")


#Inputs and outputs:
inputs = [1,0,1]
desired_outputs = [1,1,0]
matrix_neurons[:,0] = inputs

print("Initializing neuron matrix with random values...")
print(matrix_neurons)
print("")

print("Initializing random bias...")
Bias = [0.32, 0.5, 1]#[random.randint(0,100)/100 for k in range(3)]
print(Bias)
print("")


# Works fine!
def forwardPropagation(showComputation = False):
	if (showComputation == True):
		print("")
		print("Propagating forward...")

	# First layer propagation
	k = 1
	while( k <= len(Weights[:,0,0])):
		if(showComputation == True):
			print("")
			print("Computing next layer of neurons... ")
			print("")
		for i in range(len(Weights[0,0,:])):
			if(showComputation == True): 
				print("#", i, " ", Weights[k-1,i,:] , matrix_neurons[:, k-1], "  B: ", Bias[k-1])
				print("z = ", np.dot(Weights[k-1,i,:] , matrix_neurons[:, k-1]) + Bias[k-1])
				print("sigmoid(z) = ", sigmoid( np.dot(Weights[k-1,i,:] , matrix_neurons[:, k-1]) + Bias[k-1]))

				print("All neurons updated:               ")
				print(matrix_neurons)
				print("")
				print("")
				print("")
			matrix_neurons[i][k] = sigmoid( np.dot(Weights[k-1,i,:] , matrix_neurons[:, k-1]) + Bias[k-1])
		k += 1


forwardPropagation()

print("The values in the neurons are currently...")
print(matrix_neurons)


# Backpropagation algorithm
def derivative_cost_weight(layer, last_layer_neuron, current_layer_neuron):
	total_cost = Cost(matrix_neurons[:,layers-1] , desired_outputs)
	return 2 * (matrix_neurons[last_layer_neuron][layer - 1]) * sigmoid(matrix_neurons[current_layer_neuron][layer], True) * total_cost

def derivative_cost_bias(layer):
	cost = Cost(matrix_neurons[:,layers-1] , desired_outputs)
	sum = 0
	for neuron in range(len(matrix_neurons[:,layer])):
		sum += neuron
	average_value_in_layer = sum/len(matrix_neurons[:,layer])
	return 2*sigmoid(average_value_in_layer, True) * cost

def derivative_cost_neuron(layer, nNeuron):
	pass

def backpropagation(alpha = 0.01):
	# Loop that goes through all layers backwards
	for layer in reversed(range(1,layers)):
		b = derivative_cost_bias(layer)
		Bias[layer-1] = Bias[layer-1] - alpha*b
		for current_layer_neuron in range(len(matrix_neurons[:,layer])):
			print(Weights)
			print("")
			print("")
			for last_layer_neuron in range(len(matrix_neurons[:,layer-1])):
				d = derivative_cost_weight(layer, last_layer_neuron, current_layer_neuron)
				w_jk = Weights[current_layer_neuron][last_layer_neuron][layer-1]
				w_jk = w_jk - alpha * d
				Weights[last_layer_neuron][current_layer_neuron][layer-1] = w_jk
				




def training(W = Weights, matrix = matrix_neurons, b = Bias, alpha = 0.01):
	#training only the last layer

	for neuron in matrix[:, layers - 1]:
		print(neuron)
		neuron = 1



def printOutputs(output = desired_outputs):
	print("Desired outputs: ")
	print(output)
	print("")


def printBias(b = Bias):
	print("Bias: ", Bias)
	#print(Bias)
	print("")

def printNeurons(neurons = matrix_neurons):
	printBias()
	print("Neurons: ")
	forwardPropagation()
	print(neurons)
	print("Cost: ", Cost(matrix_neurons[:, 3], desired_outputs))



forwardPropagation()
#backpropagation()
training()
print(matrix_neurons)
print(Cost(matrix_neurons[:, 3], desired_outputs))







