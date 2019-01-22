# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:28:12 2018

@author: Bhaskar_Kishore
"""


import pandas as pd


data=pd.read_csv('creditcard.csv')

data=data.sample(frac=0.1,random_state=1)

#data.hist(figsize=(20,20))

Fraud=data[data['class']==1]
Valid=data[data['class']==0]

outlier_fraction=len(Fraud)/(len(Valid))
print(outlier_fraction)

print('Fraud cases:',format(len(Fraud)))
print('Valid cases:',format(len(Valid)))

columns=data.columns.tolist()

columns=[c for c in columns if c not in ['class']]

target="class"

X=data[columns]
y=data[target]

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

state =1
classifier={
        "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20,
                            contamination=outlier_fraction),
        "Isolation Forest": IsolationForest(max_samples=len(X),
                                            contamination=outlier_fraction,
                                            random_state=state)
        
                            
                                                                         
                                            }


n_outliers=len(Fraud)
for i,(clf_name,clf) in enumerate(classifier.items()):
    
    if clf_name=="Local Outlier Factor":
        y_precd=clf.fit_predict(X)
        scores_precd=clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_precd=clf.decision_function(X)
        y_precd=clf.predict(X)
        
        y_precd[y_precd==1]=0
        y_precd[y_precd==-1]=1
        
        n_errors=(y_precd !=y).sum()
        
        print(clf_name,n_errors)
        print(accuracy_score(y,y_precd))
        print(classification_report(y,y_precd))
    