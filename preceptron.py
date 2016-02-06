

# from Wikipedia mostly
def dot_product(values, weights):
	return sum(value * weight for value, weight in zip(values, weights))


# returns true if a dry run has been completed over the values
# inputs are same as optimize_weights without learning rate
def dry_run(weights, inputs, expecteds, threshold):
	error_count = 1
	count = 0
	for inps, expected in zip(inputs, expecteds):
		product = dot_product(inps, weights)
		activation = -1
		if product >= threshold:
			activation = 1
		success = (activation==expected)
		if not success:
			print("input " + str(inps) + " was not a success. " + "expected " + str(expected) + " found " + str(activation))
			return False
	return True


# parameters:
# - weights: a list of length n. Elemnts of weight are numbers that specify weight of input i.
# - inputs: a list of length m. Each element in inputs must be a list of length n.
# - expected: a  binary list (1's or 0's). The ith element in expected represents
# 	the expected outcome of the ith input in inputs.
# - threshold: the threshold that the preceptron will operate on
# 	
# Returns:
# - weights: the optimized weights that would classify the inputs.
# 
# Reference: Computational Neuroscience and Cognitive Modeling (Britt Anderson)
def optimize_weights(weights, inputs, expecteds, threshold, learning_rate):
	error_count = 1
	count = 0
	while error_count > 0 and count < 20:
		count += 1
		print('*' * 50)
		error_count = 0
		for inps, expected in zip(inputs, expecteds):
			product = dot_product(inps, weights)
			state = -1
			activation = product >= threshold
			if activation:
				state = 1
			success = (activation==expected)
			if not success:
				error_count += 1
			for index, value in enumerate(inps):
				weights[index] += state*expected*value * learning_rate
			print("input: " + str(inps) + " success: " + str(success) + " error count: " + str(error_count) + " weights:" + str(weights))
			if dry_run(weights, inputs, expecteds, threshold):
				break
	print("final weights" + str(weights))
	return weights

inp = [[0.3,0.7], [-0.6, 0.3], [0.7, 0.3], [-0.2, -0.8]]
exp = [1,-1,1,-1]
weig = [-0.6,0.8]
lear_rate = 0.5

optimize_weights(weig, inp, exp, 0, lear_rate) 






