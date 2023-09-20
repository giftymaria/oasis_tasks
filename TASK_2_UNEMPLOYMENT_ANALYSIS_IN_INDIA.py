#!/usr/bin/env python
# coding: utf-8

# # UNEMPLOYMENT ANALYSIS IN INDIA

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


df=pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
df.head()


# In[5]:


df.shape


# In[6]:


df.tail()


# In[7]:


df.info()


# In[8]:


df.describe().T


# In[9]:


df.isnull().sum()


# In[10]:


df['Region'].value_counts()


# In[11]:


df.columns


# In[12]:


df.columns=['state','date','frequency','estimated unemployment rate',
            'estimated employed','estimated labour participation rate','region','longitude','latitude']


# In[13]:


df.columns


# In[14]:


df['date']=pd.to_datetime(df['date'],dayfirst=True) #changing dtype of date


# In[15]:


df['month']=df['date'].dt.month


# In[16]:


df.head()


# In[18]:


sns.heatmap(df.corr(),annot=True)
plt.show()

DATA VISUALISATION
# In[23]:


plt.subplot(2,1,1)
plt.title('INDIAN UNEMPLOYMENT')
sns.histplot(x='estimated unemployment rate',hue='region',data=df)
plt.show()


plt.subplot(2,1,2)
sns.histplot(x='estimated employed',hue='region',data=df)
plt.show()


# In[38]:


#region wise


# In[36]:


plt.figure(figsize=(20,5))
plt.title('UNEMPLOYMENT IN INDIA -STATE WISE',size=14)
sns.barplot(x='region',y='estimated unemployment rate',data=df)
plt.xlabel('REGION',fontsize=14)
plt.ylabel('ESTIMATED UNEMPLOYMENT',fontsize=14)
plt.xticks(rotation=90, size=14)
plt.show()


# In[39]:


#statewise


# In[37]:


plt.figure(figsize=(20,5))
plt.title('UNEMPLOYMENT IN INDIA -STATE WISE',size=14)
sns.barplot(x='state',y='estimated unemployment rate',data=df)
plt.xlabel('REGION',fontsize=14)
plt.ylabel('ESTIMATED UNEMPLOYMENT',fontsize=14)
plt.xticks(rotation=90, size=14)
plt.show()


# In[45]:


plt.figure(figsize=(20,5))
G = df.groupby(['state'], as_index=False).mean()
g=G.sort_values(by='estimated unemployment rate')
sns.barplot(x='state',y='estimated unemployment rate',data=g)
plt.xlabel('REGION',fontsize=14)
plt.ylabel('ESTIMATED UNEMPLOYMENT',fontsize=14)
plt.xticks(rotation=90, size=14)
plt.show()


# # INFERENCE
MOST IMPACTED STATES ARE:
                        -Haryana
                        -Tripura
                        -Jharkand
                        -Bihar
                        -Delhi
                        -PuducherryRegion wise comparison:Highest unemployment rate is in 'NORTH' and lowest in 'WEST.'
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




