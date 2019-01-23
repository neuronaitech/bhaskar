# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:45:51 2018

@author: Bhaskar_Kishore
"""

import pandas as pd
import numpy as np


from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation



df=pd.read_csv('n3.csv')
df=df.dropna()

df = df[['Open','High','Low','Close']]

forecast_out = int(1) # predicting 30 days into future
df['Prediction'] = df[['Low']].shift(-1) #  label column with data shifted 30 units up

X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
X = X[:-forecast_out] # remove last 30 from X

y = np.array(df['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

# Training
clf = LinearRegression()
clf.fit(X_train,y_train)
# Testing
confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)