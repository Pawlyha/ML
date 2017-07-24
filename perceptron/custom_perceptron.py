import numpy as np


class CustomPerceptron():
    def __init__(self, learning_rate, steps):
        self.rate = learning_rate
        self.steps = steps
        self.errors = []
        self.weights = []

    def __heaviside(self, x, weights):
        return int(np.dot(x, weights) > 0)

    def fit(self, x, y):
        self.weights = np.zeros(x.shape[1])
        self.errors = np.zeros(x.shape[1])

        for i in range(self.steps):
            error = 0
            for xi, expected in zip(x, y):
                predicted = self.__heaviside(xi, self.weights)
                w_ = self.rate * (predicted - expected)
                self.weights += w_ * xi
                error += int(w_ != 0)

            self.errors += error

        return self
