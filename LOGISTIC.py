# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 13:59:42 2018

@author: Bhaskar_Kishore
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.metrics import classification_report

df=pd.read_csv("N1.csv")
df=df.dropna()
x=df[['Open','High','Low','Close','Adj Close','Volume']]
x=preprocessing.scale(x)

# If the tomorrow’s open price is higher than today’s open price, then we will buy the stock (1), else we will sell it (-1).
y=np.where(df['Close'].shift(1)>df['Close'],1,0)


poly = PolynomialFeatures(degree = 1)
x = poly.fit_transform(x)


cross_val = cross_val_score(LogisticRegression(),x,y,scoring='accuracy')
print('crossvalidation'+'\n',cross_val.mean())
split=int(0.7*len(df))
x_train ,x_test,y_train,y_test=x[:split],x[split:],y[:split],y[split:]

pca = PCA(n_components = 1)
x_tr = pca.fit_transform(x_train)
x_te = pca.transform(x_test)

lr=LogisticRegression()
lr=lr.fit (x_train,y_train)

print(lr.score(x_train , y_train))
print(lr.score(x_test, y_test))
#print("y_pred=",lr.predict(np.array([[296.55,297.55,293.10,296.60,296.75,295.41,1607049]]).reshape(1,7)))
print(lr.predict(poly.transform(np.array([292.00,	300.00	,291.05,	298.95,	298.95	,9080903	]).reshape(1, 6))))
predicted=lr.predict(x_test)


print('Confusion Matrix' + '\n',metrics.confusion_matrix(y_test,predicted))
print('Accuracy_Score'+'\n',accuracy_score(y_test,predicted))
print('Classification Report'+'\n',metrics.classification_report(y_test,predicted))
plt.scatter(range(len(y_test)),y_test,color='red')
plt.show()
plt.scatter(range(len(y_test)),predicted)
plt.show()


lr.fit(x_tr, y_train)
#Training set
plt.scatter(x_tr, y_train)
plt.plot(x_tr, lr.predict(x_tr))
plt.show()


# #Test set
plt.scatter(x_te, y_test)
plt.plot(x_te, lr.predict(x_te))
plt.show()