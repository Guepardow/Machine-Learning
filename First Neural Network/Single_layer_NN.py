import numpy as np
import random

# Activation function will be sigmoid
def sigmoid(x):
	return 1/(1+np.exp(-x))

def deriveSigmoid(x):
	return sigmoid(x)*(1 - sigmoid(x))

# Generate random Weights and bias
def generateWeights(nSynapses):
	Weights = []
	for i in range(nSynapses):
		Weights.append(random.randint(0, 100) / 100)
	return Weights

# Generates random bias between [0, 100]
def generateBias(nLayer):
	Bias = []
	for i in range(nLayer-1):
		Bias.append(random.randint(0,100)/100)
	return Bias

def Cost(y,output):
	return (y-output)**2

# Get random Weights and Bias for the neural network
B = generateBias(3)
W = generateWeights(2)
print(" ")
print("Generating random Bias: ", B)
print("Generating random Weights: ", W)


# Training set:
T = [[0,0],[1,1]]

input = 1
output = 0

def forwardPropagation():
	a = sigmoid(input*W[0] + B[0])
	y = sigmoid(a*W[1] + B[1])
	return a, y

a, y = forwardPropagation()

print("NN output: ", y, "   Expected output: ", output, "   Error: ", abs(output - y))

# Backpropagation algorithm:
def gradientDescent(alpha,a,y):
	W[1] = W[1] - alpha*(a*deriveSigmoid(y)*2*(y-output))
	B[1] = B[1] - alpha*(deriveSigmoid(y)*2*(y-output))
	W[0] = W[0] - alpha*(input*deriveSigmoid(a)*2*(y-output))
	B[0] = B[0] - alpha*(deriveSigmoid(a)*2*(y-output))

initial_cost = Cost(y,output)

for i in range(30):
	for j in range(100):
		gradientDescent(0.01, a, y)
		a, y = forwardPropagation()
	cost = Cost(y,output)
	#print("#", i * 100, "  Weights: ", W, " B: ", B, "    Cost: ", cost, "   Cost reduced by: ", (1 - cost/initial_cost)*100, "%")
	print("NN output: ", y, "    Cost: ", cost)
print(W)

