import numpy as np
import random
import math

dataset_unfiltered = np.array([
								[0,					0,					0,				0,				0,				15,		85],
								[0,					0,					0,				0,				1,				8,		92],
								[0,					0,					0,				1,				0,				25,		75],
								[0,					0,					0,				1,				1,				16,		84],
								[0,					0,					1,				0,				0,				30,		70],
								[0,					0,					1,				0,				1,				20,		80],
								[0,					0,					1,				1,				0,				40,		60],
								[0,					0,					1,				1,				1,				38,		62],
								[0,					1,					0,				0,				0,				50,		50],
								[0,					1,					0,				0,				1,				40,		60],
								[0,					1,					0,				1,				0,				59,		41],
								[0,					1,					0,				1,				1,				50,		50],
								[0,					1,					1,				0,				0,				45,		55],
								[0,					1,					1,				0,				1,				38,		62],
								[0,					1,					1,				1,				0,				70,		30],
								[0,					1,					1,				1,				1,				65,		35],
								[1,					0,					0,				0,				0,				40,		60],
								[1,					0,					0,				0,				1,				35,		65],
								[1,					0,					0,				1,				0,				50,		50],
								[1,					0,					0,				1,				1,				45,		55],
								[1,					0,					1,				0,				0,				57,		43],
								[1,					0,					1,				0,				1,				50,		50],
								[1,					0,					1,				1,				0,				70,		30],
								[1,					0,					1,				1,				1,				65,		35],
								[1,					1,					0,				0,				0,				40,		60],
								[1,					1,					0,				0,				1,				35,		65],
								[1,					1,					0,				1,				0,				50,		50],
								[1,					1,					0,				1,				1,				45,		55],
								[1,					1,					1,				0,				0,				80,		20],
								[1,					1,					1,				0,				1,				75,		25],
								[1,					1,					1,				1,				0,				90,		10],
								[1,					1,					1,				1,				1,				85,		15]
								])


def generate2DMatrix(m, n):
	return [ [0 for k in range(n)] for i in range(m) ]

data_set = [ [0 for k in range(6)] for i in range(3200) ]
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


for i in range(len(training_data_set)):
	inputs[i] = training_data_set[0:4]
	desired_outputs[i] = training_data_set[5]


print(len(inputs) == len(desired_outputs))













