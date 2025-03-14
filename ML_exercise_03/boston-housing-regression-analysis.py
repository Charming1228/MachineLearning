#!/usr/bin/env python
# coding: utf-8

# ## <span id="1"></span> 1. Overview

# Columns:
# - CRIM: Per capita crime rate by town
# - ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
# - INDUS: Proportion of non-retail business acres per town
# - CHAS : Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# - NOX: Nitric oxide concentration (parts per 10 million)
# - RM: Average number of rooms per dwelling
# - AGE: Proportion of owner-occupied units built prior to 1940
# - DIS: Weighted distances to five Boston employment centers
# - RAD: Index of accessibility to radial highways
# - PTRATIO: Pupil-teacher ratio by town
# - B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
# - LSTAT: Percentage of lower status of the population
# - MEDV: Median value of owner-occupied homes in $1000s

# ## <span id="2"></span> 2. Importing Libraries and Reading the Dataset

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from collections import Counter
from IPython.core.display import display, HTML
sns.set_style('darkgrid')


# In[7]:


# from sklearn.datasets import load_boston
# boston_dataset = load_boston()
# dataset = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)

import pandas as pd

# 读取本地 CSV 文件（请替换 'boston.csv' 为你的实际文件路径）
boston_df = pd.read_csv("BostonHousingData.csv")

# 确保特征列名与 `load_boston()` 一致（如果 CSV 里已经是正确的列名，就不需要修改）
feature_columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX",
                   "PTRATIO", "B", "LSTAT", "MEDV"]

# 确保数据集包含所有必要的列
boston_df = boston_df[feature_columns]

# 将 `MEDV` 作为目标变量
dataset = boston_df.copy()  # 复制原数据集，避免修改原数据
target = dataset["MEDV"]  # 目标变量
dataset = dataset.drop(columns=["MEDV"])  # 仅保留特征列

# 显示前几行数据
print(dataset.head())
print(target.head())  # 查看目标变量


# In[4]:


dataset.head()


# As you seen, there isn't "MEDV" column that we will try to predict. Let's add the column to our dataset.

# In[ ]:


# dataset['MEDV'] = boston_dataset.target


# In[6]:


dataset.head()


# ## <span id="3"></span> 3. Data Analysis

# ### <span id="4"></span> Data Preprocessing

# Are there missing values? There isn't any missing values as shown below.

# In[ ]:


dataset.isnull().sum()


# In[ ]:


X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values.reshape(-1,1)


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# 练习：在下方划分训练集和测试集, test_size=0.3
X_train, X_test, y_train, y_test = ...


# In[ ]:


print("Shape of X_train: ",X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)


# ### <span id="5"></span> Visualizing Data

# In[ ]:


corr = dataset.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(10, 10))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
#show plot
plt.show()


# In[ ]:


sns.pairplot(dataset)
plt.show()


# ## <span id="6"></span> 4. Regression Models

# ### <span id="7"></span> Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
regressor_linear = LinearRegression()
# 练习：在下方开始训练
...


# In[ ]:


from sklearn.metrics import r2_score

# Predicting Cross Validation Score the Test set results
cv_linear = cross_val_score(estimator = regressor_linear, X = X_train, y = y_train, cv = 10)

# Predicting R2 Score the Train set results
y_pred_linear_train = regressor_linear.predict(X_train)
r2_score_linear_train = r2_score(y_train, y_pred_linear_train)

# Predicting R2 Score the Test set results
y_pred_linear_test = regressor_linear.predict(X_test)
r2_score_linear_test = r2_score(y_test, y_pred_linear_test)

# Predicting RMSE the Test set results
rmse_linear = (np.sqrt(mean_squared_error(y_test, y_pred_linear_test)))
print("CV: ", cv_linear.mean())
print('R2_score (train): ', r2_score_linear_train)
print('R2_score (test): ', r2_score_linear_test)
print("RMSE: ", rmse_linear)


# ### <span id="12"></span> Decision Tree Regression

# In[ ]:


# Fitting the Decision Tree Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor_dt = DecisionTreeRegressor(random_state = 0)
# 练习：在下方开始训练
...


# In[ ]:


from sklearn.metrics import r2_score

# 练习：Predicting Cross Validation Score
cv_dt = ...

# 练习：Predicting R2 Score the Train set results
y_pred_dt_train = ...
r2_score_dt_train = ...

# 练习：Predicting R2 Score the Test set results
y_pred_dt_test = ...
r2_score_dt_test = ...

# 练习：Predicting RMSE the Test set results
rmse_dt = ...
print('CV: ', cv_dt.mean())
print('R2_score (train): ', r2_score_dt_train)
print('R2_score (test): ', r2_score_dt_test)
print("RMSE: ", rmse_dt)


# ## <span id="14"></span> 5. Measuring the Error

# In[ ]:


models = [('Linear Regression', rmse_linear, r2_score_linear_train, r2_score_linear_test, cv_linear.mean()),
          ('Decision Tree Regression', rmse_dt, r2_score_dt_train, r2_score_dt_test, cv_dt.mean()),]


# In[ ]:


predict = pd.DataFrame(data = models, columns=['Model', 'RMSE', 'R2_Score(training)', 'R2_Score(test)', 'Cross-Validation'])
predict


# ### <span id="15"></span> Visualizing Model Performance

# In[ ]:


f, axe = plt.subplots(1,1, figsize=(18,6))

predict.sort_values(by=['Cross-Validation'], ascending=False, inplace=True)

sns.barplot(x='Cross-Validation', y='Model', data = predict, ax = axe)
#axes[0].set(xlabel='Region', ylabel='Charges')
axe.set_xlabel('Cross-Validaton Score', size=16)
axe.set_ylabel('Model')
axe.set_xlim(0,1.0)
plt.show()


# In[ ]:


f, axes = plt.subplots(2,1, figsize=(14,10))

predict.sort_values(by=['R2_Score(training)'], ascending=False, inplace=True)

sns.barplot(x='R2_Score(training)', y='Model', data = predict, palette='Blues_d', ax = axes[0])
#axes[0].set(xlabel='Region', ylabel='Charges')
axes[0].set_xlabel('R2 Score (Training)', size=16)
axes[0].set_ylabel('Model')
axes[0].set_xlim(0,1.0)

predict.sort_values(by=['R2_Score(test)'], ascending=False, inplace=True)

sns.barplot(x='R2_Score(test)', y='Model', data = predict, palette='Reds_d', ax = axes[1])
#axes[0].set(xlabel='Region', ylabel='Charges')
axes[1].set_xlabel('R2 Score (Test)', size=16)
axes[1].set_ylabel('Model')
axes[1].set_xlim(0,1.0)

plt.show()


# In[ ]:


predict.sort_values(by=['RMSE'], ascending=False, inplace=True)

f, axe = plt.subplots(1,1, figsize=(18,6))
sns.barplot(x='Model', y='RMSE', data=predict, ax = axe)
axe.set_xlabel('Model', size=16)
axe.set_ylabel('RMSE', size=16)

plt.show()

