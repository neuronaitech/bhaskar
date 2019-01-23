# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 09:39:07 2018

@author: Bhaskar_Kishore
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('WIPRO2.csv',parse_dates=True,index_col='Date')  

data.info()

data['Close'].plot(grid=True,figsize=(8,8))

data['50d']=np.round(data['Close'].rolling(window=50).mean(),5)
data['250d']=np.round(data['Close'].rolling(window=250).mean(),5)

data[['Close','50d','250d']].tail()
data[['Close','50d','250d']].plot(grid=True,figsize=(8,8))

data['50d-250d']=data['50d']-data['250d']
data['50d-250d'].tail()

SD=10
data['Regime']=np.where(data['50d-250d']>SD,1,0)
data['Regime']=np.where(data['50d-250d']<SD,-1,data['Regime'])
data['Regime'].value_counts()

data['Regime'].plot(lw=1.5)
plt.ylim([-1.1,1.1])


data['Market']=np.log(data['Close']/data['Close'].shift(1))

data['strategy']=data['Regime'].shift(1)*data['Market']

data[['Market','strategy']].cumsum().apply(np.exp).plot(grid=True,figsize=(8,8))


data[['Market','strategy','Regime']].plot(subplots=True,figsize=(8,8))