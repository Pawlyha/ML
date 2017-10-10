import quandl, math
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['label'] = df['Adj. Close']

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))

current_df = df[:-forecast_out]

# will be used for for predictions
future_df = df[-forecast_out:]

X = np.array(current_df.drop('label', 1))
y = np.array(current_df['label'])

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training process
model = LinearRegression()
model.fit(X_train, y_train)

eps = model.score(X_test, y_test)

# 1.0 ????
print('Eps: ', eps)

# Prediction
X_future = np.array(future_df.drop('label', 1))
y_future_real = np.array(future_df['label'])

X_future = preprocessing.scale(X_future)

y_predicted = model.predict(X_future)

#future_df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close']].plot()
#df[['label']].plot()

print(future_df.head())


predicted_df = future_df
predicted_df['label'] = y_predicted

real_ax = future_df['label'].plot()
predicted_df['label'].plot(ax=real_ax)

plt.show()


'''
print(y_future_real)
print(y_predicted)
print(y_future_real - y_predicted)

X = np.array(df.drop(['label'], 1))[:-forecast_out]
Y = np.array(df['label'])

print(X.shape)
print(Y.shape)
'''
