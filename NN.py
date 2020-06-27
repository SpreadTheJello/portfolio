# Problem: have data of 9 flowers, with a length, width, and a color (red or blue)
# However, one of those flowers has a missing color.
# This NN will get to the closest probability for the color of the mystery flower.

import numpy as np
from matplotlib import pyplot as plt

# each point is length, width, type (0 blue, 1 red) for a flower
data = [[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5,  .5, 1],
        [2,    .5, 0],
        [5.5, 1,   1],
        [1,   1,   0]]

# this is the flower with the missing color
mystery_flower = [4.5, 1]

# neural network:
#
#    o      1 output: flower color
#   / \     w1, w2, b
#  o   o    2 inputs: length, width
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

# scatter plot of data
def scat_plot():
    plt.axis([0, 6, 0, 4])
    plt.grid()
    for i in range(len(data)):
        point = data[i]
        color = 'r'
        if point[2] == 0:
            color = 'b'
        plt.scatter(point[0], point[1], c=color)
    plt.show()

# def NN(m1, m2, w1, w2, b):
#     z = m1 * w1 + m2 * w2 + b
#     return sigmoid(z)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1-sigmoid(x))

# training
for i in range(10000):
    rIndex = np.random.randint(len(data))
    point = data[rIndex]


# def cost(b):
#     return (b - 4) ** 2

# def num_slope(b):
#     h = 0.0001
#     return (cost(b+h) - cost(b))/h

# def slope(b):
#     return 2 * (b-4)

# w1 = numpy.random.randn()
# w2 = numpy.random.randn()
# b = 8
# b = b - .1 * slope(b)

# print(NN(3, 1.5, w1, w2, b))