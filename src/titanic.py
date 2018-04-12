import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
import src.data_providers as dp
from pandas.plotting import scatter_matrix


def display_correlation(df):
    plt.matshow(df.corr())
    plt.xlabel(df.columns.values)
    plt.ylabel(df.columns.values)
    plt.show()


class AgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    @staticmethod
    def cabin_to_charcode(cabin):
        if not isinstance(cabin, str):
            return 0
        return ord(cabin[0])

    @staticmethod
    def insert_age(row):
        print(row)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ages = X.groupby('Pclass').median()['Age']

        for c, age in enumerate(ages):
            X.loc[(X['Pclass']==c+1) & (X['Age'].isnull()), 'Age'] = age

        dp.write_titanic_result(X, 'foo')
        return X

class CategorycalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    @staticmethod
    def cabin_to_charcode(cabin):
        if not isinstance(cabin, str):
            return 0
        return ord(cabin[0])

    @staticmethod
    def insert_age(row):
        print(row)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.loc[:, 'Cabin'] = X['Cabin'].apply(lambda s: self.cabin_to_charcode(s))
        X.loc[:, 'Embarked'] = X['Embarked'].apply(lambda s: self.cabin_to_charcode(s))
        X.loc[:, 'Sex'].replace({'male': 1, 'female': 0}, inplace=True)

        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']

    def transform(self, X, y=None, **fit_params):
        return X[self.features]

    def fit(self, X, y=None):
        return self


pipeline = Pipeline([
    ('selector', FeatureSelector()),
    ('age transformer', AgeTransformer()),
    ('cat transformer', CategorycalTransformer()),
    ('imputer', Imputer()),
    ('std scaler', StandardScaler())
])

# ------------ Train ------------------------
X_train = dp.load_titanic_dataset('train')

print(X_train.head(50))

y_train = X_train.loc[:,'Survived']
X_train.drop(['Survived'], axis=1, inplace=True)


X_train = pipeline.fit_transform(X_train)

# ------------ Test -------------------------

X_test = dp.load_titanic_dataset('test')

X_test = pipeline.fit_transform(X_test)

# ------------ Model Training ---------------

cls = SVC()
cls.fit(X_train, y_train)

scores = cross_val_score(cls, X_train, y_train, cv=3)
print(scores)

result = cls.predict(X_test)


# ------------ Prepare results --------------


ix_from = X_train.shape[0] + 1
ix_to = ix_from + len(result)

res_df = pd.DataFrame()
res_df['PassengerId'] = np.arange(start=ix_from, stop=ix_to, step=1)
res_df['Survived'] = result

dp.write_titanic_result(res_df)







