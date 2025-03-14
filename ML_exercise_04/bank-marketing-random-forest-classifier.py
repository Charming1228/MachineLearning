#!/usr/bin/env python
# coding: utf-8

# ## Bank Marketing 数据集 随机森林的编程实践

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# Importing the generic python libraries for Data processing and manipulation
import math
import pandas as pd
import numpy as np

# Importing libarries for spilting the train and test dataset
from sklearn.model_selection import train_test_split

# Importing Libraries for Feature Scaling
from sklearn.preprocessing import StandardScaler

# Importing libraries for Dimensionality Reduction
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA

# Importing libraries for different Classifiers for training
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn import tree
#from natsort import index_natsorted
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Importing GridSearchCV for Hyper Parameter tuning of models
from sklearn.model_selection import GridSearchCV

# Importing the Visualization libraries
import matplotlib.pyplot as plt 
import seaborn as sns
# import graphviz

# Importing Libraries for Resampling
from sklearn.utils import resample

# Importing the Performance metrics libraries
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score  
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Importing the missing number libraries
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', None)

# Importing Plotly graphic library for vizualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Importing the warnings library
import warnings
warnings.filterwarnings('ignore')

# Importing Libraries related to Handling Imbalanced Classes
from imblearn.over_sampling import SMOTE
from collections import Counter


# ## 数据预处理

# In[ ]:


# Using the read_excel function of pandas library for importing the data from excel sheet
data = pd.read_csv("D:\\25Spring\\25Spring_MachineLearning\\MachineLearning\\ML_exercise_04\\bank.csv")#这是你bank文件的路径地址
data


# In[ ]:


# Below code will segregate the different features by role and data type
metadata = []
for feature in data.columns:
    # Defining the role
    if feature == 'deposit':
        role = 'target'
    else:
        role = 'input'
         
    # Defining the level
    if 'bin' in feature or feature == 'target':
        level = 'binary'
    elif 'cat' in feature:
        level = 'nominal'
    elif data[feature].dtype == float:
        level = 'interval'
    else:
        level = 'ordinal'
        
    # Initialize keep to True for all variables except for id
    keep = True
     
    # Defining the data type 
    dtype = data[feature].dtype
    
    # Creating a Dict that contains all the metadata for the variable
    feature_dict = {
        'varname': feature,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    metadata.append(feature_dict)
    
meta1 = pd.DataFrame(metadata, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta1.set_index('varname', inplace=True)
meta1


# In[ ]:


# Below code will segregate the different features by role, category and data type
metadata = []
for feature in data.columns:
    # Defining the role
    if feature == 'deposit':
        use = 'target'
    else:
        use = 'input'
         
    # Defining the type
    if 'bin' in feature or feature == 'target':
        type = 'binary'
    elif 'cat' in feature:
        type = 'categorical'
    elif data[feature].dtype == float or isinstance(data[feature].dtype, float):
        type = 'real'
    else:
        type = 'integer'
        
    # Initialize preserve to True for all variables except for id
    preserve = True

    
    # Defining the data type 
    dtype = data[feature].dtype
    
    category = 'none'
    # Defining the category
    if 'ind' in feature:
        category = 'individual'
    elif 'reg' in feature:
        category = 'registration'
    elif 'car' in feature:
        category = 'car'
    elif 'calc' in feature:
        category = 'calculated'
    
    
    # Creating a Dict that contains all the metadata for the variable
    feature_dictionary = {
        'varname': feature,
        'use': use,
        'type': type,
        'preserve': preserve,
        'dtype': dtype,
        'category' : category
    }
    metadata.append(feature_dictionary)
    
meta2 = pd.DataFrame(metadata, columns=['varname', 'use', 'type', 'preserve', 'dtype', 'category'])
meta2.set_index('varname', inplace=True)
meta2


# In[ ]:


meta2[(meta2.type == 'categorical') & (meta2.preserve)].index


# In[ ]:


# Select the Features whose data type is Object
print(data.dtypes)
data.select_dtypes(object)


# In[ ]:


#seperating the ICU lable column
# without_deposit_column = data.drop('deposit', axis = 1)       
deposit_column = data['deposit']

#finding columns that are not of type float or int
colums_to_convert = data.select_dtypes(object).columns   
colums_to_convert


# In[ ]:


#performing hotcoding using the get_dummies method of pandas libraries
data = pd.get_dummies(data, columns = colums_to_convert)      
data.head(500)


# In[ ]:


data = data.drop('deposit_yes', axis = 1)  


# In[ ]:


data. rename(columns = {'deposit_no':'deposit'}, inplace = True)


# In[ ]:


#adding the deposit column again at the last position
       
data.head(5)


# In[ ]:


# Verify if any missing values in dataset
msno.matrix(data)

print('NaN values =', data.isnull().sum().sum())
print("""""")
vars_with_missing = []
for feature in data.columns:
    missings = data[feature].isna().sum()
    if missings > 0 :
        vars_with_missing.append(feature)
        missings_perc = missings / data.shape[0]
        print('Variable {} has {} records ({:.2%}) with missing values.'.format(feature, missings, missings_perc))
print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))


# In[ ]:


data.shape


# In[ ]:


# Check if any imbalanced classes in dataset
sns.set_style('white')
sns.set(font_scale=1)
plt.figure()
sns.countplot(x=data.deposit,palette='nipy_spectral')
plt.show()


# In[ ]:


plt1 = plt


# In[ ]:


X_data = np.array(data.drop(['deposit'], axis = 1))
Y_data = np.array(data[['deposit']])
print(X_data.shape)
print(Y_data.shape)


# In[ ]:


X_data_df = data.drop(['deposit'], axis = 1)


# ## 数据集划分

# In[ ]:


# Splitting the data into train and test dataset using train_test_split library
xtrain, xvalid, ytrain, yvalid = train_test_split(X_data, Y_data, stratify=Y_data, random_state=0, train_size=0.7)


# 练习：# 任务：按8:2划分数据集
# 调整train_size参数
# xtrain, xvalid, ytrain, yvalid =...

# In[ ]:


# Using the library StandardScaler to scale the data before training the data 
scaler = StandardScaler()


# In[ ]:


ytrain.shape


# In[ ]:


ytrain = ytrain.reshape(ytrain.size, 1)


# In[ ]:


xtrain_scaled = scaler.fit_transform(xtrain)
xvalid_scaled = scaler.transform(xvalid)


# In[ ]:


print(xtrain_scaled.shape)
print(ytrain.shape)


# In[ ]:


def ass(y_true,y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  accuracy=(tp+tn)/(tp+fp+fn+tn)
  specificity = tn/(tn+fp)
  sensitivity=tp/(tp+fn)
  print("Accuracy:",accuracy*100)
  print("Sensitivity:",sensitivity*100)
  print("Specificity:",specificity*100)
  print("ROC_AUC_Score:",roc_auc_score(y_true, y_pred)*100)
  


# In[ ]:


xtrain_scaled.shape


# In[ ]:


ytrain = ytrain.squeeze()


# In[ ]:


ytrain.shape


# In[ ]:


# Function to diaply the Confusion Matrix 混淆矩阵
def plot_cm(actual, pred):
    cm = confusion_matrix(actual, pred)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    plt1.title("Decision Tree Confusion Matrix")
    plt1.ylabel('Actual label')
    plt1.xlabel('Predicted label \n\n' + 
               "Accuracy: {:.2f}\n".format(accuracy_score(actual, pred)) + 
               "Precision: {:.2f}\n".format(precision_score(actual, pred, average='weighted')) + 
               "Recall: {:.2f}\n".format(recall_score(actual, pred, average='weighted')) +
               "F1: {:.2f}\n".format(f1_score(actual, pred, average='weighted')))


# ## 随机森林算法

# In[ ]:


# Using Random FOrest Classifier to train and predict on original dataset
RF_object = RandomForestClassifier()
RF_object.fit(xtrain_scaled, ytrain)
y_pred=RF_object.predict(xvalid_scaled)
ass(yvalid,y_pred)


# 练习：使用n_estimators=100训练随机森林
# 提示：修改RandomForestClassifier参数
# 

# In[ ]:


#练习
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# 初始化随机森林分类器（设置随机种子为42）
rf_clf = RandomForestClassifier(______=42)

# 练习1：使用训练集训练模型
# 提示：使用fit方法，注意ytrain需要转换形状
rf_clf.______(xtrain_scaled, ______.squeeze())

# 练习2：进行5折交叉验证（使用准确率作为评分标准）
# 提示：使用cross_val_score
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(
    ______, 
    ______, 
    ______, 
    cv=______,
    scoring=______
)

# 练习3：预测训练集和测试集结果
y_train_pred = rf_clf.______(______)
y_test_pred = rf_clf.______(xvalid_scaled)

# 练习4：计算评估指标
# 训练集准确率
train_acc = ______(ytrain, y_train_pred)
# 测试集准确率
test_acc = ______(yvalid, y_test_pred)
# 测试集F1分数
test_f1 = ______(yvalid, y_test_pred, average='weighted')
# 混淆矩阵
conf_matrix = ______(yvalid, y_test_pred)

print(f"训练集准确率: {train_acc:.2f}")
print(f"测试集准确率: {test_acc:.2f}") 
print(f"测试集F1分数: {test_f1:.2f}")
print("混淆矩阵:")
print(conf_matrix)


# ## 可视化

# In[ ]:


# Call the custom function to display the Confusion Matrix
plot_cm(yvalid, y_pred)


# In[ ]:




