import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC


def load_data(type="train"):
    path = "../datasets/titanic/" + type + ".csv"
    df = pd.read_csv(path, na_values=np.nan)
    return df


def display_correlation(df):
    plt.matshow(df.corr())
    plt.xlabel(df.columns.values)
    plt.ylabel(df.columns.values)
    plt.show()


class InputTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        X['Cabin'] = X['Cabin'].apply(lambda s: self.cabin_to_charcode(s))
        X['Sex'].replace({'male': 1, 'female': 0}, inplace=True)
        X.dropna(inplace=True)

        return X

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.fit_transform(X,y)

    def cabin_to_charcode(self, cabin):
        if not isinstance(cabin, str):
            return -1
        return ord(cabin[0])


df = load_data()

df = InputTransformer().fit_transform(df)

X = df[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Fare']]
y = df['Survived']


clf = SVC()
clf.fit(X, y)

df_test = load_data('test')
df_test = InputTransformer().fit_transform(df_test)

X_test = df_test[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Fare']]
y_test = df_test['Survived']

print(clf.score(X_test, y_test))


# X, y = load_data()
