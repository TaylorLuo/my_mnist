import numpy as np
import math

x = np.matrix([[0.5], [1.0]])
t = np.matrix([[1.0], [0.0]])
w1 = np.matrix([[0.1, 0.2], [0.3, 0.4]]).T
w2 = np.matrix([[0.6, 0.7], [0.8, 0.9]]).T
b1 = np.matrix([[0.5], [0.5]])
b2 = np.matrix([[1.0], [1.0]])

print('forward>>>>>>>>')

# logit1 = np.dot(w1,x)+b1
logit1 = w1 * x + b1
h1 = 1 / (1 + np.exp(-logit1))
logit2 = np.dot(w2, h1) + b2
h2 = 1 / (1 + np.exp(-logit2))
y = h2
cost = np.sum(np.power((t - y), 2) / 2)

print('x:', x)
print('w1:', w1)
print('b1:', b1)
print('w2:', w2)
print('b2:', b2)
print('logit1:', logit1)
print('h1:', h1)
print('logit2:', logit2)
print('h2:', h2)
print('cost:')
print(cost)
print('>>>>>>>>>>end forward')

print('<<<<<<<<<<backward')
delta_2 = np.multiply(np.multiply(y - t, y), (1 - y))
nabla_w2 = np.dot(delta_2, h1.T)
nabla_b2 = delta_2
sp_1 = np.multiply(h1, 1 - h1)
delta_1 = np.multiply(np.dot(w2.T, delta_2), sp_1)
nabla_w1 = np.dot(delta_1, x.T)
nabla_b1 = delta_1
new_w2 = w2 - nabla_w2
new_b2 = b2 - nabla_b2
new_w1 = w1 - nabla_w1
new_b1 = b1 - nabla_b1

print('delta_2', delta_2)
print('sp_1:', sp_1)
print('delta_1', delta_1)
print('nabla_w1', nabla_w1)
print('nabla_b1', nabla_b1)
print('nabla_w2', nabla_w2)
print('nabla_b2', nabla_b2)
print('------------end backward')
