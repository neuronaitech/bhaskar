# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 11:35:38 2018

@author: Bhaskar_Kishore
"""

import numpy as np
import pandas as pd

df=pd.read_csv('AMZND.csv')
df=df.dropna()
X=df[['Open','High','Low','Close','Volume']]

#y=np.where(df['Close'].shift(1)>df['Close'],1,0)
y=np.array([[1]])
def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative_sigmoid(x):
    return x*(1-x)

#variable intilization

epoch=5000  #traning itterations
lr=0.1      #learning rate
inputlayer_neurons=X.shape[1] # no.features in data set
hiddenlayer_neurons=3         
output_neurons=1


#weight and bias intialization

wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):
    #forward propagation
    hiddenlayer_input1=np.dot(X,wh)
    hiddenlayer_input=hiddenlayer_input1+bh
    hiddenlayer_activation=sigmoid(hiddenlayer_input)
    outputlayer_input1=np.dot(hiddenlayer_activation,wout)
    outputlayer_input=outputlayer_input1+bout
    output=sigmoid(outputlayer_input)
    
    #backward
    E=y-output
    slope_output_layer=derivative_sigmoid(output)
    slope_hidden_layer=derivative_sigmoid(hiddenlayer_activation)
    d_output=E*slope_output_layer
    Error_at_hidden_layer=d_output.dot(wout.T)
    d_hidden_layer=Error_at_hidden_layer*slope_hidden_layer
    wout+=hiddenlayer_activation.T.dot(d_output)*lr
    bout+=np.sum(d_output,axis=0,keepdims=True)*lr
    wh += X.T.dot(d_hidden_layer) *lr
    bh += np.sum(d_hidden_layer, axis=0,keepdims=True) *lr
    
print(output)