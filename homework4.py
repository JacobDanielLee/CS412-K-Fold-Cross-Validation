#!/usr/bin/env python
# coding: utf-8

# In[40]:


import sys
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np


def get_splits(n, k):
    r = []
    for i in range(0,n):
        r.append(i)
    r = shuffle(r)
    
    d = [[] for i in range(k)]
    count = 0
    for i in r:
        index = count % k
        d[index].append(i)
        count += 1
            
    return d


# In[47]:


def my_cross_val(method, X, y, k):
    a = []
    
    if method == "LinearSVC":
        model_t = LinearSVC(max_iter = 2000)
    elif method == "RandomForestClassifier":
        model_t = RandomForestClassifier(max_depth=20, random_state=0, n_estimators=500)
    elif method == "LogisticRegression":
        model_t = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial')
    elif method == "SVC":
        model_t = SVC(gamma='scale', C = 10)
    elif method == "XGBClassifier":
        model_t = XGBClassifier(max_depth=5)
    else:
        return
    
    ki_list = get_splits(len(X), k)
    
    index_list = []
    for i in ki_list:
        for j in i:
            index_list.append(j)
    
    for i in ki_list:
        train_X = []
        test_X = []
        train_y = []
        test_y = []
        for index in index_list:
            if index in i:
                test_X.append(X[index])
                test_y.append(y[index])
            else:
                train_X.append(X[index])
                train_y.append(y[index])
        model_t.fit(train_X,train_y)
        test_result = model_t.predict(test_X)
        wrong = []
        for j in range(0,len(test_result)):
            if (round(test_result[j], 4) != round(test_y[j], 4)):
                wrong.append(1)
        k_error = len(wrong) / len(test_result)
        a.append(k_error)
    
    return a


# In[48]:


def my_train_test(method, X, y, p, k):
    a = []
    
    if method == "LinearSVC":
        model_t = LinearSVC(max_iter = 2000)
    elif method == "RandomForestClassifier":
        model_t = RandomForestClassifier(max_depth=20, random_state=0, n_estimators=500)
    elif method == "LogisticRegression":
        model_t = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial')
    elif method == "SVC":
        model_t = SVC(gamma='scale', C = 10)
    elif method == "XGBClassifier":
        model_t = XGBClassifier(max_depth=5)
    else:
        return
    
    while k > 0:
        #create random indices
        r = []
        for i in range(0,len(X)):
            r.append(i)
            r = shuffle(r)
        
        train_X = []
        test_X = []
        train_y = []
        test_y = []
        split_index = len(X) * p
        split_index = round(split_index)
    
        for i in range(0, len(X)):
            if (i <= split_index):
                train_X.append(X[r[i]])
                train_y.append(y[r[i]])
            if (i > split_index):
                test_X.append(X[r[i]])
                test_y.append(y[r[i]])
        model_t.fit(train_X,train_y)
        test_result = model_t.predict(test_X)
        wrong = []
        for j in range(0,len(test_result)):
            if (round(test_result[j], 4) != round(test_y[j], 4)):
                wrong.append(1)
        k_error = len(wrong) / len(test_result)
        a.append(k_error)
        k = k-1
    
    return a


# In[38]:


print(get_splits(10, 2))


# In[ ]:




