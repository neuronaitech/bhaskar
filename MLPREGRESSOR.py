# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 02:02:09 2018

@author: Bhaskar_Kishore
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing,cross_validation
from sklearn.neural_network import MLPRegressor
import logging
import sklearn
from sklearn2pmml import sklearn2pmml, PMMLPipeline

df=pd.read_csv('TCS.csv',index_col='Date').tail(100)

#del df['Series']
#del df['Symbol']
df=df.dropna()
#df2=pd.read_csv('DOWT.csv',index_col='Date').tail(100)
#df2=df2.dropna()
#
#df=pd.concat([df1,df2],axis=1)

forecast_out = int(5) # predicting 30 days into future
df['Prediction'] = df[['Open']].shift(-1) #  label column with data shifted 30 units up

X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
X = X[:-forecast_out] # remove last 30 from X

y = np.array(df['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

#clf=MLPRegressor(hidden_layer_sizes=(10),activation='relu',solver='adam',)

#clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
#		                    hidden_layer_sizes=(50), random_state=1, max_iter=1000)

#pipeline = PMMLPipeline([
#		  ('clf', clf)
#		])
#pipeline.fit(df[df.columns.difference(["Open"])], df["Prediction"])			
#sklearn2pmml(pipeline, "PredictiveModel.pmml")

estimator = MLPRegressor(
        hidden_layer_sizes=(1000,) * 4,
        activation='relu',
        max_iter=int(1e2),
        verbose=True,
        random_state=1,
        tol=0)
logging.info('Estimator: {}'.format(estimator.get_params()))

y_pred = sklearn.model_selection.cross_val_predict(
        estimator=estimator,
        X=X,
        y=y,
        cv=6
    )
estimator.fit(X_train,y_train)

confidence = estimator.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = estimator.predict(X_forecast)
print(forecast_prediction)

z=y_pred.tolist()
price1 = pd.concat([pd.Series(df['Open']),pd.Series(z)], ignore_index = True, copy = True)
print(price1.tail())
fig, ax = plt.subplots(1, 1)
price1.plot(ax=ax,color='k',label='actual')
price1.iloc[99:].plot(ax=ax,color='r',label='predict')
plt.xlabel('index')
plt.ylabel('Close price') 
plt.title('actual & forecasted close')
plt.legend()
plt.show()


actual = pd.read_csv('TCSACTUAL1.csv')
plt.plot(actual['Open'][:5],color='k',label='actual')
plt.plot(z[:5],color='red',label='predicted') 
plt.legend()
plt.title("actualVsForecasted")
plt.show()