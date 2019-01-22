# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 00:06:07 2018

@author: BHASKARA KISHORE
"""

import pandas as pd
import numpy as np


from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



#df=pd.read_csv('n3.csv')
#df=df.dropna()
#
#df = df[['Open','High','Low','Close']]
data1 = pd.read_csv('wheat-2013-supervised.csv')
data2 = pd.read_csv('wheat-2014-supervised.csv')

#print(data1.head())
#merge two dataset for train the model later
merged = data1.append(data2, ignore_index=True)
merged = merged[["Latitude","Longitude","apparentTemperatureMax","apparentTemperatureMin","cloudCover","dewPoint","humidity","precipIntensity","precipIntensityMax","precipProbability","precipAccumulation","precipTypeIsRain","precipTypeIsSnow","precipTypeIsOther",	"pressure",	"temperatureMax","temperatureMin","visibility",	"windBearing","windSpeed","NDVI","DayInSeason","Yield" ]]
merged.to_csv('merged.csv', index=None, header=True)
mg = pd.read_csv('merged.csv')

#del mg['Date']
#del mg['State']

forecast_out = int(1) # predicting 30 days into future
mg['Prediction'] = mg[['Yield']].shift(-1) #  label column with data shifted 30 units up

X = np.array(mg.drop(['Prediction'], 1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
X = X[:-forecast_out] # remove last 30 from X

y = np.array(mg['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Training
clf = LinearRegression()
mg.fillna(mg.mean(),inplace=True)
np.isnan(mg.values.any())
clf.fit(X_train,y_train)
# Testing
confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)