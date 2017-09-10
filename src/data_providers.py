import numpy as np


def load_iris_dataset():
    path = '../datasets/iris/iris_dataset.csv'
    cols = np.genfromtxt(path, delimiter=',', max_rows=1, dtype=None)
    data = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=[0, 1, 2, 3])
    labels = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=[4], dtype=np.str)

    return cols, data, labels

