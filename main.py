from perceptron import custom_perceptron as my
import numpy as np

p = my.CustomPerceptron(0.8, 10)

data = np.genfromtxt('data/iris_dataset.csv', delimiter=',', names=True, dtype=None)
d = np.matrix(data)

[print() for xi in data]
