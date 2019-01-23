# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:45:51 2018

@author: Bhaskar_Kishore
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation,svm
from sklearn.model_selection import learning_curve
from sklearn.svm import SVR
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt




df=pd.read_csv('wiproweek1.csv')
df=df.dropna()

df = df[['Open','High','Low','Close','Volume']]

forecast_out = int(2) # predicting 1 days into future
df['Prediction'] = df[['Close']].shift(-1) #  label column with data shifted 1 units up

X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:] # set X_forecast equal to last 1
X = X[:-forecast_out] # remove last 1 from X

y = np.array(df['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)
#train_scores, valid_scores = validation_curve(svm.SVR()), X, y, "alpha",
#                                               np.logspace(-7, 3, 3))
# Training
clf = svm.SVR(kernel='linear')
clf.fit(X_train,y_train)
# Testing
confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)


forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)

#plt.plot(train_scores)
#plt.show()
#plt.plot(valid_scores)
