# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 21:05:09 2018

@author: Bhaskar_Kishore
"""


from sklearn import linear_model

from sklearn import datasets
iris=datasets.load_iris()
X=iris.data
y=iris.target

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=5)
# Create linear regression object
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(X_train, y_train)
linear.score(X_train, y_train)
#Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
####


from sklearn.linear_model import LogisticRegression

# Create logistic regression object
model = LogisticRegression()
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)

#Predict Output
predicted= model.predict(X_test)



#KNN

from sklearn.neighbors import KNeighborsClassifier

# Create KNeighbors classifier object model 
KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
# Train the model using the training sets and check score
model.fit(X, y)

#Predict Output
predicted= model.predict(X_test)
 

##KMEANS
#Import Library
from sklearn.cluster import KMeans

# Create KNeighbors classifier object model 
k_means = KMeans(n_clusters=3, random_state=0)
# Train the model using the training sets and check score
model.fit(X,y)
#Predict Output
predicted= model.predict(X_test)


