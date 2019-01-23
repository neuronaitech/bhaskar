# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 14:16:54 2018

@author: Bhaskar_Kishore
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

d=pd.read_csv('WIPRO1.csv').tail(10)
d=d.dropna()

X=pd.DataFrame(d[['Open','High','Low']])
y=pd.DataFrame(d['Close'])

#p=PCA(1).fit_transform(X)


#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state = 7)
model = DecisionTreeRegressor() 
#model=LinearRegression()
model.fit(X,y)
p1=model.predict(X)

#print(model.predict(279.90))
#print(model.score(X_test,y_test))
#print(model.score(X_train,y_train))
#print(model.predict(np.array([278.00,282.80,277.50]).reshape(-1, 3)))


p=PolynomialFeatures(degree=4)
X_poly=p.fit_transform(X)
p.fit(X_poly,y)
#model2=LinearRegression()
model2=DecisionTreeRegressor()
model2.fit(X_poly,y)


#plt.scatter(X['Open'],y)
#plt.plot(X,model.predict(X))
#plt.show()


#plt.scatter(X['Open'],y)
plt.plot(X,model2.predict(p.fit_transform(X)))
plt.show()

#train
#plt.scatter(X_train,y_train)
#plt.scatter(X_train,model.predict(X_train),color='k')
#plt.show()
#
##test
#plt.scatter(X_test,y_test)
#plt.scatter(X_test,p,color='y')
#plt.show()

