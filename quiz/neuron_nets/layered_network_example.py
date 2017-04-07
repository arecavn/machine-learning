# Layered network calculation

import numpy as np
# Hidden layer weights
w_h1_1 = np.array([1, 1, -5]).reshape(1, 3)
w_h1_2 = np.array([3, -4, 2]).reshape(1, 3)

# We can concatenate the matrices to form 2 x 3
weights = np.concatenate((w_h1_1, w_h1_2), axis=0)

# Output weights
w_o = np.array([2, -1])

# Input
inp = np.array([1, 2, 3])

# Shapes to understand which matrix to reshape
print('weight scalar 1', w_h1_1.shape)
print('weight scalar 2', w_h1_2.shape)
print('output scalar', w_o.shape)
print('input scalar', inp.shape)
print('overall weights matrix', weights.shape)

# matrix of ndoes
# (2, 3) . (3 x 1) would give (2 x 1)
nodes_matrix = np.dot(weights, inp).reshape(1, 2)
print('Nodes matrix', nodes_matrix.shape)

# (1 x 2), nodes
# multiplied by (2 x 1), output
# would give (1 x 1)
print np.dot(nodes_matrix, w_o)
