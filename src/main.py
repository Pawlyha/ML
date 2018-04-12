import src.data_providers as dp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)



def show_digit(x_row):
    digit = x_row.reshape(28, 28)
    plt.imshow(digit, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()


def shuffle_data(X, y):
    index = np.random.permutation(len(y))
    return X[index], y[index]

train_c = 60000
test_c = 10000

X, y = dp.load_mnist_dataset()
X, y = shuffle_data(X, y)

X_train, X_test, y_train, y_test = X[:train_c], X[train_c:train_c + test_c], y[:train_c], y[train_c:train_c + test_c]



print(y_train[0:100])





'''
kn_clf = KNeighborsClassifier()

grid_search_params = [{'n_neighbors': [3, 4, 5], 'weights': ['uniform', 'distance'], 'algorithm': ['ball_tree', 'kd_tree']}]

grid_search = GridSearchCV(kn_clf, grid_search_params, cv=3)

grid_search.fit(X_train, y_train)

gs_scores = grid_search.cv_results_

print(grid_search.best_params_)

for k, val in gs_scores:
    print(k, val)
'''



'''

'''

kn_clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
scores = cross_val_score(kn_clf, X_train, y_train, cv=5, scoring="accuracy")
print(scores)



#show_digit(36000)
