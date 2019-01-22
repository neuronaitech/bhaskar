# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 08:57:19 2018

@author: Bhaskar_Kishore
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random
#from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import cross_val_score
#from sklearn.model_selection import train_test_split
df2 = pd.read_csv ("APPLE1.csv",index_col='Date').tail(100)
#df2=pd.DataFrame(df2[['Open Price','High Price','Low Price','Close Price']]).dropna()
df1 =pd.read_csv("DOWT.csv",index_col='Date').tail(100)
#df1=pd.DataFrame(df1[['Open','High','Low','Close']]).dropna()
#del df2['Symbol']
#del df2['Series']
df=pd.concat([df2,df1],axis=1)
#how many data we will use 
# (should not be more than dataset length )
data_to_use= 100

# number of training data
# should be less than data_to_use
train_end =80
 
 
total_data=len(df)
 
#most recent data is in the end 
#so need offset
start=total_data - data_to_use
 
#currently doing prediction only for 1 step ahead
steps_to_predict =1
 
yclose = df.iloc [start:total_data ,3].rolling(window=3).mean().dropna()   #Close price
ylow = df.iloc [start:total_data ,2].rolling(window=3).mean().dropna()    #low CLOSE
yopen = df.iloc [start:total_data ,0] .rolling(window=3).mean().dropna()   #open
yhigh = df.iloc [start:total_data ,1] .rolling(window=3).mean().dropna()  #high
#vturnover = df.iloc [start:total_data ,9] .rolling(window=6).mean().dropna()   # TURN OVER
ydclose = df.iloc [start:total_data ,9].rolling(window=3).mean().dropna()   #Close price
ydopen = df.iloc [start:total_data ,6].rolling(window=3).mean().dropna()    #open CLOSE
ydhigh = df.iloc [start:total_data ,7] .rolling(window=3).mean().dropna()   #high
ydlow = df.iloc [start:total_data ,8] .rolling(window=3).mean().dropna()  #low
#vturnover = df.iloc [start:total_data ,9] .rolling(window=6).mean().dropna()  
 
print ("yclose head :")
print (yclose.head())
 
yt_ = yclose.shift (-1)
yt1_=ydclose.shift(-1)
     
data = pd.concat ([yclose, yt_, ydclose,yt1_, yopen, yhigh, ylow,ydopen,ydhigh,ydlow], axis =1)
data. columns = ['yclose', 'yt_', 'ydclose','yt1_', 'yopen', 'yhigh', 'ylow','ydopen','ydhigh','ydlow']
     
data = data.dropna()
     
y = data ['yt_']      

cols =['yclose',     'yopen', 'yhigh', 'ylow','ydclose','ydopen','ydhigh','ydlow']
x = data [cols]

#from sklearn.preprocessing import PolynomialFeatures
#poly = PolynomialFeatures(2)
#poly.fit_transform(x)

scaler_x = preprocessing.MinMaxScaler ( feature_range =( -1, 1))
x = np. array (x).reshape ((len( x) ,len(cols)))
x = scaler_x.fit_transform (x)
 
    
scaler_y = preprocessing. MinMaxScaler ( feature_range =( -1, 1))
y = np.array (y).reshape ((len( y), 1))
y = scaler_y.fit_transform (y)
#x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)
     
x_train = x [0: train_end,]
x_test = x[ train_end +1:len(x),]    
y_train = y [0: train_end] 
y_test = y[ train_end +1:len(y)]  
x_train = x_train.reshape (x_train. shape + (1,)) 
x_test = x_test.reshape (x_test. shape + (1,))
#x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=0)


from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers import  Dropout
from keras.models import load_model
import tensorflow as tf
#from tensorflow.contrib.rnn import BasicLSTMCell,MultiRNNCell
np.random.seed(7)
random.random()

fit = Sequential ()
fit.add (LSTM (  100,forget_bias_init='one', activation = 'tanh', input_shape =(len(cols), 1) ,unit_forget_bias=True,return_sequences=True ))
#fit.add(LSTM(5))
fit.add(Dropout(0.2))
fit.add(LSTM(100))
fit.add(Dropout(0.2))
#fit.add (Dense (output_dim =100, activation = 'linear'))
#fit.add (Dense (output_dim 100, activation = 'linear'))
fit.add (Dense (output_dim =50, activation = 'linear'))
fit.add (Dense (output_dim =50, activation = 'linear'))
fit.add (Dense (output_dim =25, activation = 'linear'))
fit.add (Dense (output_dim =25, activation = 'linear'))
fit.add (Dense (output_dim =12, activation = 'linear'))
fit.add (Dense (output_dim =6, activation = 'linear'))
#fit.add (Dense (output_dim =60, activation = 'linear'))

fit.add (Dense (output_dim =1, activation = 'linear'))
 
fit.compile (loss ="mean_squared_error" , optimizer = "adam")   
fit.fit (x_train, y_train, batch_size =100, nb_epoch =20,validation_data=(x_test, y_test), verbose=1, shuffle=False)
 
#print (fit.summary())

#rnn_cell = MultiRNNCell([BasicLSTMCell(15),BasicLSTMCell(15)])
# 
#fit = tf.keras.models.Sequential([
#  tf.keras.layers.Flatten(),
#  tf.keras.layers.Dense(512, activation=tf.nn.relu),
#  tf.keras.layers.Dropout(0.2),
#  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#])
#fit.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])
#
#fit.fit(x_train, y_train, epochs=5)
#fit.evaluate(x_test, y_test)

score_train = fit.evaluate (x_train, y_train, batch_size =1)
score_test = fit.evaluate (x_test, y_test, batch_size =1)
print (" in train MSE = ", round( score_train ,4)) 
print (" in test MSE = ", score_test )
    
#pred1 = fit1.predict (x_test) 
pred1=fit.predict(x_test)
pred1 = scaler_y.inverse_transform (np. array (pred1). reshape ((len( pred1), 1)))

prediction_data =np.flip( pred1[-4:].reshape(1,4)[0],axis=0)
for i in range(len(prediction_data)):
    print(prediction_data[i])
print ("prediction data:")
print (prediction_data)
#pred = pred1[-25:]
  
#import json
#from keras.models import  load_model

## Option 1: Save Weights + Architecture
#fit.save_weights('model_weights.h5')
#with open('model_architecture.json', 'w') as f:
#    f.write(fit.to_json())
## Option 1: Load Weights + Architecture
#with open('model_architecture.json', 'r') as f:
#    new_model_1 = model_from_json(f.read())
#new_model_1.load_weights('model_weights.h5')

#print(pred)   
z=prediction_data.tolist()
price1 = pd.concat([pd.Series(data['yclose']),pd.Series(z)], ignore_index = True, copy = True)
print(price1.tail())
fig, ax = plt.subplots(1, 1)
price1.plot(ax=ax,color='k',label='actual')
price1.iloc[95:].plot(ax=ax,color='r',label='predict')
plt.xlabel('index')
plt.ylabel('Close price') 
plt.title('actual & forecasted close')
plt.legend()
plt.show()


#from keras.models import load_model
#fit.save('ls7.h5')
#
#
#model = load_model('ls7.h5')
#
#
#lstmweights=model.get_weights()
#fit.set_weights(lstmweights)
#model_train=model.evaluate(x_train,y_train)
#model_test=model.evaluate(x_test,y_test)
#print(model_train)
#print(model_test)
# 

#load using numpy
#np.save('lstm',prediction_data)
##
#k=np.load('lstm.npy')
##
#print(k)
##
###$text format
#np.savetxt('lstm.txt',prediction_data)
#np.loadtxt('lstm.txt')



actual = pd.read_csv('APPLEACTUAL.csv')
actual=actual.dropna()
actual=actual['Close']
plt.plot(actual[:5],color='k',label='actual')
plt.plot(prediction_data[:5],color='red',label='predicted') 
plt.legend()
plt.title("actualVsForecasted")
plt.show()

