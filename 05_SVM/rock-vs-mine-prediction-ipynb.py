#!/usr/bin/env python
# coding: utf-8

# import libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# data collection and data processing

# In[ ]:


#loading dataset to pandas dataframe


sonar_data= pd.read_csv('data/sonar_data.csv',header = None)


# In[ ]:


sonar_data.head()
# 61 columns, the last one is the target (R: rock or M: mine)


# In[ ]:


# number of rows and columns
sonar_data.shape


# In[ ]:


sonar_data.describe()


# In[ ]:


sonar_data[60].value_counts()


# M--->Mine
# 
# R--->Rock

# In[ ]:


sonar_data.groupby(60).mean()


# In[ ]:


#separating data and lebals
x= sonar_data.drop(columns=60, axis=1)
y= sonar_data[60]
y = pd.get_dummies(y,drop_first=True)


# In[ ]:


x.iloc[:,:3]


# In[ ]:


plt.figure(figsize =( 20,20))
sns.heatmap(x.iloc[:,:10].corr(),annot = True)


# In[ ]:


print(x.shape)
print(y.shape)


# training and test data

# In[ ]:


x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.1,stratify=y,random_state=1)


# In[ ]:


print(x.shape, x_train.shape, x_test.shape, y_test.shape)


# In[ ]:


print(x_train)
print(y_train)


# model training--->SVM

# In[ ]:


# 练习：使用SVM进行训练与测试，并计算指标
model = ... # 定义模型
... # 训练
... # 测试
... # 计算正确率
print(f'accuracy: {...}')


# In[ ]:


# 练习：使用线性模型进行训练与测试，并计算指标
model = ... # 定义模型
... # 训练
... # 测试
... # 计算正确率
print(f'accuracy: {...}')


# In[ ]:


# 练习：使用随机森林进行训练与测试，并计算指标
model = ... # 定义模型
... # 训练
... # 测试
... # 计算正确率
print(f'accuracy: {...}')

