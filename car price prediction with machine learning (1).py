#!/usr/bin/env python
# coding: utf-8
name : Prajakta Ramesh Chavan
Date:20/02/2024
Domain: Data Science
Task_03:oasis Infobyte: car price prediction with machine learning.
# In[27]:


#libray install
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# In[28]:


#importing the data
data = pd.read_csv(r"C:\Users\Dell\Desktop\car data.csv")
data


# In[29]:


# dataset name and output 5 mentioned:
car_data.head(5)


# In[30]:


data.info()


# In[33]:


data.describe()


# In[34]:


data.shape


# In[35]:


data.isnull().sum()


# In[36]:


data.tail()


# In[37]:


print(data.Fuel_Type.value_counts())


# In[38]:


print(data.Selling_type.value_counts())


# In[39]:


print(data.Transmission.value_counts())


# In[80]:


import datetime
x=datetime.datetime.now()
print(x)


# In[41]:


data.head(5)


# In[42]:


import seaborn as sns
sns.boxplot(data['Selling_Price'])


# In[43]:


sorted(data['Selling_Price'],reverse=True)


# In[44]:


data[(data['Selling_Price']>=33.0) &(data['Selling_Price']<=35.0)]


# In[45]:


data=data[~(data['Selling_Price']>=33.0) &(data['Selling_Price']<=35.0)]


# In[46]:


data.shape


# In[47]:


data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)


# In[48]:


data.replace({'Selling_type':{'Dealer':0,'Individual':1}},inplace=True)


# In[49]:


data.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)


# In[50]:


data.head()


# In[51]:


data.tail()


# In[52]:


x=data.drop(['Car_Name','Selling_Price'],axis=1)


# In[53]:


y=data['Selling_Price']


# In[54]:


print(x)


# In[55]:


print(y)


# In[56]:


from sklearn.model_selection import train_test_split


# In[57]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[58]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[59]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


# In[60]:


#! pip install XGBoost


# In[61]:


lr=LinearRegression()
lr.fit(x_train,y_train)

rf=RandomForestRegressor()
rf.fit(x_train,y_train)

xgb=GradientBoostingRegressor()
xgb.fit(x_train,y_train)

xg=XGBRegressor()
xg.fit(x_train,y_train)


# In[62]:


y_pred_1=lr.predict(x_test)
y_pred_2=rf.predict(x_test)
y_pred_3=xgb.predict(x_test)
y_pred_4=xg.predict(x_test)


# In[63]:


from sklearn import metrics
score1=metrics.r2_score(y_test,y_pred_1)
score2=metrics.r2_score(y_test,y_pred_2)
score3=metrics.r2_score(y_test,y_pred_3)
score4=metrics.r2_score(y_test,y_pred_4)


# In[64]:


print(score1,score2,score3,score4)


# In[65]:


final_data=pd.DataFrame({'models':['LR','RF','XGB','XG'],"score":[score1,score2,score3,score4]})


# In[66]:


final_data


# In[67]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x=final_data['models'],y=final_data['score'],data=final_data, palette = 'magma')


# In[68]:


xg = XGBRegressor()
xg_final = xg.fit(x,y)


# In[69]:


import joblib


# In[70]:


fig=plt.figure(figsize=(7,5))
plt.title('correlation between present price and selling price')
sns.regplot(x='Present_Price',y='Selling_Price',data=car_data)


# In[71]:


from sklearn.preprocessing import StandardScaler
Scaler=StandardScaler()


# In[72]:


#!pip install standardScaler


# In[73]:


x_train=Scaler.fit_transform(x_train)
x_test=Scaler.transform(x_test)


# In[74]:


model=LinearRegression()


# In[75]:


model.fit(x_train,y_train)


# In[76]:


pred=model.predict(x_test)


# In[77]:


from sklearn.metrics import  mean_absolute_error,mean_squared_error,r2_score


# In[78]:


print("MAE: ",(metrics.mean_absolute_error(pred,y_test)))
print("MSE: ",(metrics.mean_squared_error(pred,y_test)))
print("R2 score: ",(metrics.r2_score(pred,y_test)))


# In[79]:


sns.regplot(x=pred,y=y_test)
plt.xlabel("predicted Price")
plt.ylabel("Actual Price")
plt.title("Actual Price vs Predicted Price")
plt.show()


# In[2]:





# In[ ]:





# In[ ]:





# In[ ]:




