#!/usr/bin/env python
# coding: utf-8
 Train a machine learning model that can learn from the measurements of the iris species and classify them.
# In[55]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[56]:


from sklearn.neighbors import KNeighborsClassifier


# In[57]:


df_iris=pd.read_csv("iris.csv")

df_iris.head()


# In[58]:


df_iris.shape 


# In[59]:


df_iris=df_iris.drop(columns=['Id'])


# In[60]:


df_iris.head()


# In[61]:


df_iris.info()


# In[62]:


df_iris.describe()


# In[63]:


df_iris.isnull().sum()


# -encoding

# In[64]:


df_iris['Species'].unique()


# In[65]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_iris['Species'] = le.fit_transform(df_iris['Species'])
df_iris['Species']


# In[66]:


df_iris['Species'].unique()

EXPLORATORY DATA ANALYSIS
# In[67]:


df_iris['SepalLengthCm'].hist()
plt.title('SEPAL LENGTH')
plt.show()


# In[68]:


df_iris['SepalWidthCm'].hist()
plt.title('SEPAL WIDTH')
plt.show()

The above histograms show it is normal distribution.
# In[69]:


df_iris['PetalLengthCm'].hist()
plt.title('PETAL LENGTH')
plt.show()


# In[70]:


df_iris['PetalWidthCm'].hist()
plt.title('PETAL WIDTH')
plt.show()


# In[ ]:





# In[74]:


plt.scatter(df_iris['SepalLengthCm'], df_iris['SepalWidthCm'], c=df_iris['Species'], cmap='viridis')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Scatterplot of Sepal Length vs Sepal Width')
plt.show()


# In[75]:


plt.scatter(df_iris['PetalLengthCm'], df_iris['PetalWidthCm'], c=df_iris['Species'], cmap='viridis')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Scatterplot of Petal Length vs Petal Width')
plt.show()


# In[76]:


plt.scatter(df_iris['SepalLengthCm'], df_iris['PetalWidthCm'], c=df_iris['Species'], cmap='viridis')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Scatterplot of Sepal Length vs Petal Width')
plt.show()


# In[79]:


plt.figure(figsize=(12,6))
corr=df_iris.corr()
sns.heatmap(corr,annot=True)


# In[ ]:


-Species is highly in positive correlation with PetalWidthCm, PetalLengthCm and SepalLenghthCm .
-Species is negative in correlation with SepalWidthCm.

Model Building
# In[80]:


from sklearn.model_selection import train_test_split


# In[81]:


x=df_iris.drop(columns=['Species'])
y=df_iris['Species']


# In[82]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


# In[83]:


x_train.head()


# In[84]:


y_train.head()

LOGISTIC REGRESSION
# In[85]:


from sklearn.linear_model import LogisticRegression


# In[86]:


model=LogisticRegression()


# In[87]:


model.fit(x_train,y_train)


# In[88]:


print("Accuracy: ",model.score(x_test,y_test)*100)

DECISION TREE CLASSIFIER
# In[93]:


from sklearn.tree import DecisionTreeClassifier


# In[94]:


model=DecisionTreeClassifier()


# In[95]:


model.fit(x_train,y_train)


# In[97]:


print("Accuracy: ",model.score(x_test,y_test)*100)


# In[ ]:


KNN


# In[89]:


from sklearn.neighbors import KNeighborsClassifier


# In[90]:


model=KNeighborsClassifier()


# In[91]:


model.fit(x_train,y_train)


# In[92]:


print("Accuracy: ",model.score(x_test,y_test)*100)

************************************************************************************************************************