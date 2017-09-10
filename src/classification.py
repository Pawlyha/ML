import numpy as np
import src.data_providers as dp
import matplotlib.pyplot as plt


class ClassificationAlg:
    def __init__(self, n_steps, step):
        self.n = n_steps
        self.step = step
        self.b, self.m = 0, 0
        self.x, self.y, self.w = np.array([]), np.array([]), np.array([])

    def predict(self, x):
        z = self._z(x)
        return self.sigmoid(z)

    def train(self, x, y):
        self.x = x
        self.y = y
        self.m = self.y.shape[0]
        self.w = np.zeros(self.x.shape[1])
        self.b = 0
        errors = np.zeros(self.n)

        for i in range(self.n):
            self.w -= self.step * self._dw()
            self.b -= self.step * self._db()

            cost = self._calc_err()
            errors[i] = cost

        return self, errors

    def _z(self, x):
        z = np.dot(self.w, x.T) + self.b
        return z

    # activation function
    @staticmethod
    def sigmoid(z):
        f = 1 / (1 + np.exp(-z))
        return f

    def _dw(self):
        z = self._z(self.x)
        dw = np.dot(self.x.T, self.sigmoid(z) - self.y) / self.m
        return dw

    def _db(self):
        z = self._z(self.x)
        db = np.sum(self.sigmoid(z) - self.y) / self.m
        return db

    def _calc_err(self):
        z = self._z(self.x)
        sqr_err = np.sum(np.power(self.sigmoid(z) - self.y, 2)) / 2 * self.m
        return sqr_err



# data loading
cols, data, labels = dp.load_iris_dataset()
x = data
y = np.where(labels == 'Iris-setosa', 1, 0)
n = 10
step = 0.01

# training
alg = ClassificationAlg(n, step)
self, err = alg.train(x, y)

# final graphs
#plt.plot(np.sum(data[:, [0, 1]], axis=1), np.sum(data[:, [2, 3]], axis=1), 'ro')

plt.plot(range(n), err, 'o')
plt.show()

#print(self.predict(np.array([5.5,2.6,4.4,1.2])))

