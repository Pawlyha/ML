import quandl, math
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_Change'] = df['Adj. High'] / df['Adj. Low']
df['CO_Change'] = df['Adj. Open'] / df['Adj. Close']

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df['Adj. Close'].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))[:-forecast_out]
Y = np.array(df['label'])

print(X.shape)
print(Y.shape)


X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
