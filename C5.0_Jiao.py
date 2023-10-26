#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


# ---------------------------------------------------
# 匯入資料
columns_name = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']

data_train = pd.read_csv('adult.data', names=columns_name, header=None)
df_train = pd.DataFrame(data_train)

data_test = pd.read_csv('adult.test', names=columns_name, skiprows=1, header=None)
df_test = pd.DataFrame(data_test)

# ---------------------------------------------------
# missing value
columns_to_check = ['workclass', 'occupation', 'native-country']

df_train.replace(" ?", np.nan, inplace=True)
column_modes = df_train[columns_to_check].mode().iloc[0]
df_train.fillna(column_modes, inplace=True)

df_test.replace(" ?", np.nan, inplace=True)
column_modes = df_test[columns_to_check].mode().iloc[0]
df_test.fillna(column_modes, inplace=True)

# ---------------------------------------------------
# 將 income 轉換為 int64 (>50K:1；<=50K:0)

df_train.replace(" >50K", 1, inplace = True)
df_train.replace(" <=50K", 0, inplace = True)

df_test.replace(" >50K.", 1, inplace = True)
df_test.replace(" <=50K.", 0, inplace = True)

# ---------------------------------------------------
# Outlier

def detect_and_handle_Outlier(df, column_name, treshold=1.5):
    #IQR = Q3-Q1
    IQR = np.percentile(df[column_name],75) - np.percentile(df[column_name],25)
    #upper_outlier = Q3 + treshold*IQR 
    df=df[df[column_name] < np.percentile(df[column_name],75)+treshold*IQR]
    #lower_outlier = Q1 - treshold*IQR 
    df=df[df[column_name] > np.percentile(df[column_name],25)-treshold*IQR]
    return df

columns_to_check = ['age','education-num','hours-per-week','income']

for column in columns_to_check:
    df_train = detect_and_handle_Outlier(df_train, column)
    
for column in columns_to_check:
    df_test = detect_and_handle_Outlier(df_test, column)


# In[ ]:


# ---------------------------------------------------
# combine train and test data
df_data = pd.concat([df_train, df_test], axis=0)

# ---------------------------------------------------
# 移除無關屬性的欄位
# df_data.describe()
df_data.drop('fnlwgt', axis = 1, inplace = True)
df_data.drop('capital-gain', axis = 1, inplace = True)
df_data.drop('capital-loss', axis = 1, inplace = True)

# ---------------------------------------------------
# One Hot Encoding
df_data = pd.get_dummies(df_data)

# ---------------------------------------------------
# split train and test data
df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]

# ---------------------------------------------------
# split feature and class

train_x = df_train.drop('income', axis=1)
train_y = df_train['income']

test_x = df_test.drop('income', axis=1)
test_y = df_test['income']

train_x = train_x.astype(int)
test_x = test_x.astype(int)


# In[ ]:


# ---------------------------------------------------
# rpy2
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# ---------------------------------------------------
# 設定 C5.0 套件
C50 = importr("C50")
pandas2ri.activate()
C5_0 = robjects.r('C5.0')

# ---------------------------------------------------
# 將 DataFrame 轉換為 R 的 factor
r_train_x = pandas2ri.py2rpy(train_x)
r_train_y = pandas2ri.py2rpy(train_y)
r_train_y = robjects.r('as.factor')(r_train_y)

r_test_x = pandas2ri.py2rpy(test_x)
r_test_y = pandas2ri.py2rpy(test_y)
r_test_y = robjects.r('as.factor')(r_test_y)

# ---------------------------------------------------
# 建立模型 C5.0
model = C50.C5_0(r_train_x, r_train_y)

# ---------------------------------------------------
# 預測、評估模型好壞
train_pred = C50.predict_C5_0(model, r_train_x)
train_pred = pd.DataFrame(train_pred)

train_pred[train_pred==1] = 0
train_pred[train_pred==2] = 1

# 輸出混淆矩陣，顯示準確率
print("Train: 輸出混淆矩陣，顯示準確率")
print('====================================================================')
print(confusion_matrix(train_y, train_pred))
print(classification_report(train_y, train_pred))

train_result = df_train[['income']].copy()
train_result['predict'] = train_pred

# train_result.to_excel('C50_train_result.xlsx', index=False)

# 預測，評估模型好壞
test_pred = C50.predict_C5_0(model, r_test_x)
test_pred = pd.DataFrame(test_pred)

test_pred[test_pred==1] = 0
test_pred[test_pred==2] = 1

# 輸出混淆矩陣，顯示準確率：使用測試資料
print("Test: 輸出混淆矩陣，顯示準確率")
print('====================================================================')
print(confusion_matrix(test_y, test_pred))
print(classification_report(test_y, test_pred))


# ---------------------------------------------------
# 輸出 excel

file_path = "test_result.xlsx"

if os.path.exists(file_path):
    test_result = pd.read_excel('test_result.xlsx') # 讀取
    test_result['C5.0_pred'] = test_pred #新資料
    test_result.to_excel('test_result.xlsx', index=False) #儲存

else:
    test_result = df_test[['income']].copy()
    test_result = pd.DataFrame(test_result)
    test_result['C5.0_pred'] = test_pred #新資料
    test_result.to_excel('test_result.xlsx', index=False)


# In[ ]:




