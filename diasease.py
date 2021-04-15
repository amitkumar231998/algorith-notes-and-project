# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:18:28 2021

@author: amit kumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv(r'C:\Users\pankaj\Downloads\coaching\dtree\Datasets_DTRF\Diabetes.csv')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data.columns
data[' Class variable']=le.fit_transform(data[' Class variable'])
x=data.iloc[:,0:8]
y=data[' Class variable']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

from sklearn.tree import DecisionTreeClassifier as DT
model = DT(criterion = 'entropy')
model.fit(x_train,y_train)

pred=model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test) 
pd.crosstab(pred,y_test)

pred_train=model.predict(x_train)
accuracy_score(pred_train,y_train)
pd.crosstab(pred_train,y_train)

# second method
accuracy_test_lap = np.mean(pred == y_test)
accuracy_test_lap

accuracy_test_lap_1=np.mean(pred_train == y_train)
accuracy_test_lap_1