# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 16:43:20 2018

@author: Bhaskar_Kishore
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
#from datetime import datetime
df = pd.read_csv('techmweek.csv')
df['Date']=pd.to_datetime(df['Date'],infer_datetime_format=True)
df=df[['Date','Close']]
indxds=df.set_index(['Date'])
indxds.head(5)
plt.xlabel('Date')
plt.ylabel("Close")
plt.plot(indxds)
rolmean = indxds['Close'].rolling(window=12).mean()
rolstd = indxds['Close'].rolling(window=12).std()
print(rolmean, rolstd)
#plotting rolling statistics
original = plt.plot(indxds['Close'],color='b',label = 'original')
mean = plt.plot(rolmean,color='r',label = 'rolling mean')
std = plt.plot(rolstd,color='black',label = 'rolling std')
#plt.legend(loc='best')
plt.title('rolling mean & rolling std')
plt.show()
def dickeytest(testdata):
     print('Results of Dickey-Fuller test:')
     dftest = adfuller(testdata, autolag='AIC')
     dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
     for key, value in dftest[4].items():
          dfoutput['Critical Value ({})'.format(key)] = value
     print(dfoutput)
     
dickeytest(indxds['Close'])
#estimating trend
indxds_logscale=np.log(indxds['Close'])
plt.plot(indxds_logscale)
movingAverage = indxds_logscale.rolling(window=12).mean()
movingSTD =indxds_logscale .rolling(window=12).std()
plt.plot(indxds_logscale)
plt.plot(movingAverage,color='b')
plt.plot(movingSTD,color='g')

LogScaleMinusmovingAverage =indxds_logscale - movingAverage
LogScaleMinusmovingAverage.head(10) 
# remove nan values
LogScaleMinusmovingAverage.dropna(inplace=True)
LogScaleMinusmovingAverage.head(10)
dickeytest(LogScaleMinusmovingAverage)
diffshift=indxds_logscale-indxds_logscale.shift()
plt.plot(diffshift)
diffshift.dropna(inplace=True)
rolmean = diffshift.rolling(window=12).mean()
rolstd = diffshift.rolling(window=12).std()
print(rolmean, rolstd)
#plotting rolling statistics
original = plt.plot(diffshift,color='b',label = 'original')
mean = plt.plot(rolmean,color='r',label = 'rolling mean')
std = plt.plot(rolstd,color='black',label = 'rolling std')
#plt.legend(loc='best')
plt.title('rolling mean & rolling std')
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(diffshift, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(diffshift, lags=40,ax=ax2)
plt.show()
#arima
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(indxds_logscale, order=(3,1,0))  
results = model.fit(disp=0)
print(results.summary())  
plt.plot(diffshift,color='g')
plt.plot(results.fittedvalues, color='b')
#plt.title('RSS: %.4f'% sum((results.fittedvalues-diffshift['Close'])**2))
####
x=results.forecast(steps=15)[0]
fore=np.exp(x)
z = fore.tolist()
price1 = pd.concat([pd.Series(df['Close']),pd.Series(z)], ignore_index = True, copy = True)
print(price1.tail())
fig, ax = plt.subplots(1, 1)
price1.plot(ax=ax,color='k',label='actual')
price1.iloc[53:].plot(ax=ax,color='r',label='forecasted')
plt.xlabel('index')
plt.ylabel('Close price') 
plt.title('actual & forecasted close')
plt.legend()