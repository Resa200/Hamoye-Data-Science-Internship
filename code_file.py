#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


url="https://github.com/WalePhenomenon/climate_change/blob/master/fuel_ferc1.csv?raw=true"
fuel_data=pd.read_csv(url,error_bad_lines=False)
fuel_data.describe(include="all")


# In[3]:


fuel_data.isnull().sum()


# In[4]:


fuel_data.groupby('fuel_unit')['fuel_unit'].count()


# In[5]:


fuel_data[['fuel_unit']]=fuel_data[['fuel_unit']].fillna(value='mcf')


# In[6]:


fuel_data.isnull().sum()


# In[7]:


fuel_data.groupby('report_year')['report_year'].count()


# In[9]:


fuel_data.groupby('fuel_type_code_pudl').first()


# In[10]:


fuel_data.duplicated().any()


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


import seaborn as sns


# In[13]:


fuel_data.skew(axis=0,skipna=True)


# In[14]:


fuel_data.kurtosis()


# In[15]:


df=pd.pivot_table(fuel_data,values='fuel_cost_per_unit_burned',index='report_year')


# In[16]:


df


# In[17]:


fuel_data.corr(method='pearson')


# In[ ]:




