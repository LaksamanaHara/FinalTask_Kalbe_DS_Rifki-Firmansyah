#!/usr/bin/env python
# coding: utf-8

# ---
# Install Library
# ---

# In[1]:


#Library
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot


# In[2]:


#Baca Data#
df_customer = pd.read_csv(r"C:\Users\laksa\Documents\Project Kalbe\Customer.csv")
df_product = pd.read_csv(r"C:\Users\laksa\Documents\Project Kalbe\Product.csv")
df_store = pd.read_csv(r"C:\Users\laksa\Documents\Project Kalbe\Store.csv")
df_transaction = pd.read_csv(r"C:\Users\laksa\Documents\Project Kalbe\Transaction.csv")


# In[3]:


#Melihat Baris & Kolom Masing-masing#
df_customer.shape, df_product.shape, df_store.shape, df_transaction.shape


# ---
# Data Cleansing
# ---

# In[4]:


df_customer.head()


# In[5]:


df_product.head()


# In[6]:


df_store.head()


# In[7]:


df_transaction.head()


# In[8]:


#data cleansing df_customer
df_customer['Income'] = df_customer['Income'].replace('[,]','.',regex=True).astype('float')


# In[9]:


#data cleansing df store
df_store['Latitude'] = df_store['Latitude'].replace('[,]','.',regex=True).astype('float')
df_store['Longitude'] = df_store['Longitude'].replace('[,]','.',regex=True).astype('float')


# Gabung Data
# ---

# In[10]:


df_merge = pd.merge(df_transaction,df_customer, on=['CustomerID'])
df_merge = pd.merge(df_merge, df_product.drop(columns=['Price']), on=['ProductID'])
df_merge = pd.merge(df_merge, df_store, on=['StoreID'])


# In[11]:


df_merge.head()


# In[13]:


df_merge.to_excel('kalbe.xlsx')


# Model machine learning regresi
# ---

# In[91]:


df_regresi = df_merge.groupby(['Date']).agg({
    'Qty' : 'sum'
}).reset_index()


# In[92]:


df_regresi


# In[93]:


decomposed = seasonal_decompose(df_regresi.set_index('Date'))

plt.figure(figsize=(8,8))

plt.subplot(311)
decomposed.trend.plot(ax=plt.gca())
plt.title('Trend')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonality')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residuals')

plt.tight_layout()


# In[94]:


cut_off = round(df_regresi.shape[0] * 0.9)
df_train = df_regresi[:cut_off]
df_test = df_regresi[cut_off:].reset_index(drop=True)
df_train.shape, df_test.shape


# In[95]:


df_train


# In[96]:


df_test


# In[97]:


plt.figure(figsize=(20,5))
sns.lineplot(data=df_train, x=df_train['Date'], y=df_train['Qty']);
sns.lineplot(data=df_test, x=df_test['Date'], y=df_test['Qty']);


# In[98]:


autocorrelation_plot(df_regresi['Qty']);


# In[59]:


def rmse(y_actual, y_pred):
    """
    function to calculate RMSE
    """
    
    print(f'RMSE value {mean_squared_error(y_actual, y_pred)**0.5}')
    
def eval(y_actual, y_pred):
    """
    functional to eval machine learning modelling
    """
    
    rmse(y_actual, y_pred)
    print(f'MAE value {mean_absolute_error(y_actual, y_pred)}')


# In[99]:


#Arima#
df_train = df_train.set_index('Date')
df_test = df_test.set_index('Date')

y = df_train['Qty']

ARIMAModel = ARIMA(y, order=(40, 2, 1))
ARIMAModel = ARIMAModel.fit()

y_pred = ARIMAModel.get_forecast(len(df_test))

y_pred_df = y_pred.conf_int()
y_pred_df['predictions'] = ARIMAModel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
y_pred_df.index = df_test.index
y_pred_out = y_pred_df['predictions']

# Evaluasi performa model
mse = mean_squared_error(df_test['Qty'], y_pred_out)
mae = mean_absolute_error(df_test['Qty'], y_pred_out)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

plt.figure(figsize=(20, 5))
plt.plot(df_train['Qty'], label='Training Data')
plt.plot(df_test['Qty'], color='red', label='Test Data')
plt.plot(y_pred_out, color='black', label='ARIMA Predictions')
plt.legend()
plt.show()


# ---
# Clustering
# ---

# In[102]:


df_merge.head()


# In[105]:


df_cluster = df_merge.groupby(['CustomerID']).agg({
    'TransactionID' : 'count',
    'Qty' : 'sum',
    'TotalAmount' : 'sum'
}).reset_index()


# In[106]:


df_cluster.head()


# In[107]:


df_cluster


# In[108]:


data_cluster = df_cluster.drop(columns=['CustomerID'])
data_cluster_normalize = preprocessing.normalize(data_cluster)


# In[109]:


data_cluster_normalize


# In[115]:


K = range(2,8)
fits =[]
score = []

for k in K:
    model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(data_cluster_normalize)
    fits.append(model)
    score.append(silhouette_score(data_cluster_normalize, model.labels_, metric='euclidean'))


# In[116]:


#choose 4 cluster
sns.lineplot(x = K, y = score);


# In[118]:


df_cluster['cluster_label'] = fits[2].labels_


# In[120]:


df_cluster.groupby(['cluster_label']).agg({
    'CustomerID' : 'count',
    'TransactionID' : 'mean',
    'Qty' : 'mean',
    'TotalAmount' : 'mean'
})


# In[ ]:




