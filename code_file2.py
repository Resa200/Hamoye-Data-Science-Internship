#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[25]:


df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv')


# In[26]:


df.head()


# In[27]:


sample_linear_reg_df=df[['T2','T6']].sample(n=15,random_state=2)


# In[28]:


df.info()


# In[29]:


df=df.drop(['date'],axis=1)


# In[30]:


df=df.drop(['lights'],axis=1)


# In[31]:


from sklearn.preprocessing import MinMaxScaler


# In[32]:


scaler = MinMaxScaler()
normalised_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
features_df = normalised_df.drop(['T2','T6'],axis=1)
heating_target = normalised_df['T2']


# In[33]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features_df, heating_target, test_size=0.3, random_state=1)


# In[34]:


from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()

linear_model.fit(x_train, y_train)

predicted_values = linear_model.predict(x_test)


# In[35]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predicted_values)
round(mae, 2)


# In[36]:


from sklearn.metrics import r2_score
r2_score = r2_score(y_test, predicted_values)
round(r2_score, 2)


# In[37]:


import numpy as np
rss = np.sum(np.square(y_test - predicted_values))
round(rss, 2)


# In[38]:


from sklearn.metrics import  mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rmse, 3) 


# In[39]:


def get_weights_df(model, feat, col_name):
  #this function returns the weight of every feature
  weights = pd.Series(model.coef_, feat.columns).sort_values()
  weights_df = pd.DataFrame(weights).reset_index()
  weights_df.columns = ['Features', col_name]
  weights_df[col_name].round(3)
  return weights_df


# In[40]:


from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(x_train, y_train)


# In[41]:


from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.4)
ridge_reg.fit(x_train, y_train)


# In[42]:


linear_model_weights = get_weights_df(linear_model, x_train, 'Linear_Model_Weight')
ridge_weights_df = get_weights_df(ridge_reg, x_train, 'Ridge_Weight')
lasso_weights_df = get_weights_df(lasso_reg, x_train, 'Lasso_weight')

final_weights = pd.merge(linear_model_weights, ridge_weights_df, on='Features')
final_weights = pd.merge(final_weights, lasso_weights_df, on='Features')


# In[43]:


final_weights


# In[44]:


lasso_model = Lasso().fit(x_train,y_train)


# In[45]:


y_pred = lasso_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:




