# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 00:13:31 2023

@author: HP
"""

# importing libraries  
import numpy as nm    
import pandas as pd  
 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  
from sklearn import metrics

#importing datasets  
data_set= pd.read_csv('F:/Crop_recommendation.csv')  

x=data_set[['N','P','K','temperature','humidity','ph','rainfall']]
y=data_set[['label']]

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=3)

rfr= RandomForestClassifier(n_estimators=20, random_state=0)
rfr.fit(x_train, y_train.values.ravel())
#Predicting the test set result  
y_pred= rfr.predict(x_test)  

x= metrics.accuracy_score(y_test.values, y_pred)


print("Accuracy for Random Forest is : ",x)
#print("Accuracy for Random Forest is : ",x)

