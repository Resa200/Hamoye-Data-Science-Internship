#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv(r"C:\Users\USER\Downloads\household_power_consumption\household_power_consumption.txt",delimiter=';', low_memory=False, header=0, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'], na_values='?')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.replace("?", np.nan, inplace=True)


# In[6]:


df.isnull().sum()


# In[7]:


df = df.fillna(df.mean())
df.isnull().sum()


# In[8]:


import matplotlib.pyplot as plt
for i in range(len(df.columns)):
    plt.subplot(len(df.columns), 1, i+1)
    name = df.columns[i]
    plt.plot(df[name])
    plt.title(name, y=0)
plt.show()
plt.figure(figsize=(10,6))


# In[9]:


years = ['2007','2008','2009','2010']
plt.figure()
for i in range(len(years)):
    ax = plt.subplot(len(years), 1, i+1)
    year = years[i]
    result = df[str(year)]
    plt.plot(result['Global_active_power'])
    plt.title(str(year), y=0, loc='left')
plt.show()


# In[10]:


months = [x for x in range(1, 13)]
plt.figure(figsize=(10,6))
for i in range(len(months)):
    ax = plt.subplot(len(months), 1, i+1)
    month= '2007-' + str(months[i])
    result = df[month]
    plt.plot(result['Global_active_power'])
    plt.title(month, y=0, loc='left')
plt.show()
plt.tight_layout()


# In[11]:


plt.figure(figsize=(10,6))
plt.plot(df.Global_active_power)
plt.grid()


# In[12]:


plt.figure(figsize=(10,6))
plt.plot(df.Global_reactive_power)
plt.grid()


# In[13]:


plt.figure(figsize=(10,6))
plt.plot(df.Voltage)
plt.grid()


# In[14]:


plt.figure(figsize=(10,6))
plt.plot(df.Global_intensity)
plt.grid()


# In[15]:


plt.figure(figsize=(10,6))
plt.plot(df.Sub_metering_1)
plt.grid()


# In[16]:


plt.figure(figsize=(10,6))
plt.plot(df.Sub_metering_2)
plt.grid()


# In[17]:


plt.figure(figsize=(10,6))
plt.plot(df.Sub_metering_3)
plt.grid()


# In[18]:


import statsmodels.api as sm
from pylab import rcParams
rcParams['figure.figsize'] = 15,8


# In[19]:


decompose_series = sm.tsa.seasonal_decompose(df['Voltage'], model='additive', freq=(60*24*30))
decompose_series.plot()
plt.show()


# In[20]:


df.head()


# In[21]:


df_daily = df.resample('D').sum()
df_daily.head()


# In[22]:


df_monthly = df.resample('M').mean()
df_monthly.head()


# In[23]:


import statsmodels.api as sm
from pylab import rcParams

rcParams['figure.figsize'] = 15,8
decompose_series = sm.tsa.seasonal_decompose(df['Global_active_power'], model='additive', freq= (60*24*30))
decompose_series.plot()
plt.show()


# In[24]:


get_ipython().system('pip install --upgrade plotly')


# In[25]:


get_ipython().system('pip install pystan')
import pystan


# In[26]:


from fbprophet import Prophet


# In[27]:


df_daily.head(2)


# In[28]:


df_daily2= df_daily.reset_index()
df_daily2.head()


# In[29]:


df_daily2 = df_daily2[['datetime','Global_active_power']]
new_daily_df= df_daily2.rename(columns={"datetime":"ds","Global_active_power":"y"})


# In[ ]:


model_2 = Prophet()
model_2.fit(new_daily_df)
future = model_2.make_future_dataframe(periods=365, freq='D')
forecast2 = model_2.predict(future)
forecast2.head(5)


# In[ ]:


forecast2[['ds','yhat','yhat_lower','yhat_upper','trend','trend_lower','trend_upper']]


# In[ ]:


model_2.plot(forecast2)
plt.show()


# In[ ]:


plt.plot(df_daily.Global_active_power)
plt.show()


# In[ ]:


df['Voltage'].corr(df['Global_active_power'])


# In[ ]:


df['Global_active_power'].corr(df['Global_reactive_power'])


# In[ ]:


df['Global_reactive_power'].corr(df['Global_active_power'])


# In[ ]:


df_daily.shape


# In[ ]:


df_GAP = df_daily2['Global_active_power']


# In[ ]:


train_df = df_daily2.drop(df_daily2.index[-365:])
print(train_df.tail())


# In[ ]:


df_daily2.tail(2)


# In[ ]:


model = Prophet()
model.fit(new_train_df)


# In[ ]:


future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


y_true = df_daily2['Global_active_power'][-365:].values
y_pred = forecast['yhat'][-365:].values


# In[ ]:


MAE = np.mean(np.abs((y_true - y_pred)/y_true))
MAE


# In[ ]:


y_true.shape


# In[ ]:


forecast['yhat'].tail()


# In[ ]:


df_daily2['Global_active_power'].tail()


# In[ ]:


from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
round(rmse, 3)


# In[ ]:


model.plot_components(forecast)

