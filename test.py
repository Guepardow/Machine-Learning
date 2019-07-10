import numpy as np
def sigmoid(x, derivative = False):
	if(derivative == True):
		s = sigmoid(x,False)
		return s*(1 - s)
	return 1/(1+np.exp(-x))


while(True):
	x = float(input("Value of x: "))
	print(sigmoid(x))