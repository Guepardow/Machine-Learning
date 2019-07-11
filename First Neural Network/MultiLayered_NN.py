import numpy as np
import random
import matplotlib.pyplot as plt

# Activation function will be sigmoid
def sigmoid(x, derivative = False):
	if(derivative == True):
		s = sigmoid(x,False)
		return s*(1 - s)
	return 1/(1+np.exp(-x))

def deriveSigmoid(x):
	return sigmoid(x)*(1 - sigmoid(x))

def backward_sigmoid(z):
	return 0-(np.log( (1/z) - 1))

def accuracy_output(nn_output, desired_output_):
	a1 = ( abs(nn_output[0]-desired_output_[0]))
	a2 = (abs(nn_output[1]-desired_output_[1]))
	a3 = ( abs(nn_output[2]-desired_output_[2]))
	return (a1+a2+a3)/3, a1, a2, a3


def Cost(last_layer_neurons_vector, desired_output_vector):
	cost = 0
	for i in range(len(last_layer_neurons_vector)):
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


def generateRandomWeights(mat_neurons):
	Layers = len(mat_neurons[0])
	synapses_layers = Layers - 1
	Weights = generate3DMatrix(len(mat_neurons[:,0]), len(mat_neurons[:,0]), synapses_layers)
	return Layers, synapses_layers, Weights



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
inputs = [1,1,0]
desired_outputs = [1, 0, 0]
matrix_neurons[:,0] = inputs

print("Initializing neuron matrix with random values...")
print(matrix_neurons)
print("")

print("Initializing random bias...")
Bias = [0.32, 0.5, 1] 					#[random.randint(0,100)/100 for k in range(3)]
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
print("")


# Backpropagation algorithm
def derivative_cost_weight(layer, last_layer_neuron, current_layer_neuron):
	total_cost = Cost(matrix_neurons[:,layers-1] , desired_outputs)
	#z = backward_sigmoid(matrix_neurons[current_layer_neuron][layer])
	z = matrix_neurons[current_layer_neuron][layer]*(1 - matrix_neurons[current_layer_neuron][layer])
	return 2 * (matrix_neurons[last_layer_neuron][layer - 1]) * z * total_cost

def derivative_cost_bias(layer):
	cost = Cost(matrix_neurons[:,layers-1] , desired_outputs)
	sum = 0
	for neuron in range(len(matrix_neurons[:,layer])):
		sum += neuron
	average_value_in_layer = sum/len(matrix_neurons[:,layer])
	#z = backward_sigmoid(average_value_in_layer)
	z = average_value_in_layer*(1-average_value_in_layer)
	return 2 * z * cost


def backpropagation(alpha = 0.01):
	# Loop that goes through all layers backwards
	for layer in reversed(range(1,layers)):
		Wb = derivative_cost_bias(layer)
		Bias[layer-1] = Bias[layer-1] - alpha*Wb
		for current_layer_neuron in range(len(matrix_neurons[:,layer])):
			for last_layer_neuron in range(len(matrix_neurons[:,layer-1])):
				dW = derivative_cost_weight(layer, last_layer_neuron, current_layer_neuron)
				w_jk = Weights[layer-1][current_layer_neuron][last_layer_neuron]
				w_jk = w_jk - alpha * dW
				Weights[layer-1][current_layer_neuron][last_layer_neuron] = w_jk
				




def training(W = Weights, matrix = matrix_neurons, b = Bias, alpha = 0.01):
	#training only the last layer
	Last_layer_neuron_Position = 0
	Current_layer_neuron_position = 0
	
	for current_layer_neuron in matrix[:, layers - 1]:
		z = backward_sigmoid(current_layer_neuron)
		for last_layer_neuron in matrix_neurons[:, layers - 2]:

			dW = 2*last_layer_neuron*deriveSigmoid(z)*Cost(matrix_neurons[:, 3], desired_outputs)

			w_jk = Weights[layers-2][Current_layer_neuron_position][Last_layer_neuron_Position]

			w_jk = w_jk - alpha*dW
			Weights[layers-2][Current_layer_neuron_position][Last_layer_neuron_Position] = w_jk

			Current_layer_neuron_position += 1
			if(Current_layer_neuron_position == 3):
				Current_layer_neuron_position = 0

		Last_layer_neuron_Position = (1+Last_layer_neuron_Position)%3



newCost = Cost(matrix_neurons[:, 3], desired_outputs)
prevCost = Cost(matrix_neurons[:, 3], desired_outputs) + 1

J_history = []
out_1 = []
out_2 = []
out_3 = []

#for i in range(5000):
while(prevCost >= newCost):
	forwardPropagation()
	backpropagation()
	prevCost = newCost
	forwardPropagation()
	newCost = Cost(matrix_neurons[:, 3], desired_outputs)

	out_1.append(matrix_neurons[0][3])
	out_2.append(matrix_neurons[1][3])
	out_3.append(matrix_neurons[2][3])

	J_history.append(newCost)


print("Output layer and the desired outputs")
print(matrix_neurons[:,3] , "   " ,desired_outputs)


x = np.linspace(0,len(J_history),len(J_history))

plt.figure()

# J_History
plt.subplot(2,2,1)
plt.plot(x, J_history)
plt.title('J_history')
plt.ylabel("Error")


# Output 1 history
plt.subplot(2,2,2)
plt.plot(x, out_1)
plt.title('Output 1 history')


# Output 1 history
plt.subplot(2,2,3)
plt.plot(x, out_2)
plt.title('Output 2 history')


# Output 1 history
plt.subplot(2,2,4)
plt.plot(x, out_3)
plt.title('Output 3 history')


plt.show()


