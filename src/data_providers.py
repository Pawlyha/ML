import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata


def load_iris_dataset():
    path = '../datasets/iris.csv'
    cols = np.genfromtxt(path, delimiter=',', max_rows=1, dtype=None)
    data = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=[0, 1, 2, 3])
    labels = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=[4], dtype=np.str)

    return cols, data, labels


def load_mnist_dataset():
    mnist = fetch_mldata('MNIST original')
    return mnist['data'], mnist['target']


def load_titanic_dataset(type='train'):
    path = "../datasets/titanic/" + type + ".csv"
    df = pd.read_csv(path, na_values=np.nan)
    df.set_index('PassengerId')
    return df

def write_titanic_result(df, filename='result'):
    path = "../datasets/titanic/"+ filename +".csv"
    df.to_csv(path, index=False)




