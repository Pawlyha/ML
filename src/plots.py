
import src.data_providers as dp
import matplotlib.pyplot as plt
import numpy as np

def iris_dataset():
    cols, data, labels = dp.load_iris_dataset()
    plt.plot(np.sum(data[:, [0, 1]], axis=1), np.sum(data[:, [2, 3]], axis=1), 'ro')
    plt.show()
    #print(data)



iris_dataset()