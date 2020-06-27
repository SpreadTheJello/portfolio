# Problem: have data of 9 flowers, with a length, width, and a color (red or blue)
# However, one of those flowers has a missing color.
# This NN will get to the closest probability for the color of the mystery flower.

import os
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

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1-sigmoid(x))

# training
learnRate = 0.2
costs = []

for i in range(50000):
    rIndex = np.random.randint(len(data))
    point = data[rIndex]

    z = point[0] * w1 + point[1] * w2 + b
    prediction = sigmoid(z)

    target = point[2]
    cost = np.square(prediction - target)

    deriv_costprediction = 2 * (prediction - target)
    derivPrediction_derivZ = sigmoid_deriv(z)
    derivZ_derivW1 = point[0]
    derivZ_derivW2 = point[1]
    derivZ_derivB = 1

    derivCost_derivW1 = deriv_costprediction * derivPrediction_derivZ * derivZ_derivW1
    derivCost_derivW2 = deriv_costprediction * derivPrediction_derivZ * derivZ_derivW2
    derivCost_derivB = deriv_costprediction * derivPrediction_derivZ * derivZ_derivB

    w1 = w1 - learnRate * derivCost_derivW1
    w2 = w2 - learnRate * derivCost_derivW2
    b = b - learnRate * derivCost_derivB
    
    if i % 100 == 0:
        costSum = 0
        for j in range(len(data)):
            point = data[rIndex]

            z = point[0] * w1 + point[1] * w2 + b
            prediction = sigmoid(z)

            target = point[2]
            costSum += np.square(prediction - target)
        costs.append(costSum/len(data))

# plt.plot(costs)
# plt.show()
    
z = mystery_flower[0] * w1 + mystery_flower[1] * w2 + b
prediction = sigmoid(z)

def solve(l, w):
    phrases = ["I think it is ", "my guess is ", "my predictions show ", "the flowers color is "]
    ri = np.random.randint(len(phrases))

    z = l * w1 + w * w2 + b
    pred = sigmoid(z)
    if pred < .5:
        os.system("say " + phrases[ri] + "blue")
    else:
        os.system("say " + phrases[ri] + "red")

print("Hello!")
length = float(input("Enter a length: "))
width = float(input("Enter a width: "))
solve(length, width)

