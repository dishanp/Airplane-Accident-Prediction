#!/usr/bin/env python
# coding: utf-8

# ## Airplane Accident Prediction✈️
# ##### By Dishan Purkayastha

# Importing Required Packages:

# In[22]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder


# Importing The Dataset:

# In[3]:


df=pd.read_csv('C:\\Users\devdi\Documents\DISHAN\Airplane-Accident-master\Airplane-Accident-master\AirplaneAccident.csv')


# In[4]:


df.head()


# Elementory Analysis Of Dataset:

# In[7]:


flight = df.groupby('Severity')[['Accident_ID']].count()
flight =flight.sort_values('Accident_ID', ascending=False).reset_index()
flight.rename(columns = {'Accident_ID':'Accidents'},inplace=True)
flight


# In[8]:


labels = list(flight.Severity)
plt.figure(figsize=(10,10))
plt.title("Severity Wise Distribution",fontweight='bold',fontsize=15)
plt.tick_params(labelsize=40)
plt.pie(flight.Accidents,labels=labels,textprops={'fontsize': 13});
plt.savefig('sev.png', dpi=300)


# In[10]:


plt.figure(figsize=(20,10))
plt.xlabel('Severity Level')
plt.ylabel('Accidents')
plt.title('Severity wise distribution');
plt.bar(flight.Severity,flight.Accidents);
plt.savefig('sev2.png', dpi=300)


# In[11]:


df.tail(3)


# In[13]:


flight = df.groupby('Total_Safety_Complaints')[['Accident_ID']].count()
flight =flight.sort_values('Accident_ID', ascending=False).reset_index()
flight.rename(columns = {'Accident_ID':'Accidents'},inplace=True)
flight


# In[14]:


labels = list(flight.Total_Safety_Complaints)
plt.figure(figsize=(10,10))
plt.title("Severity Wise Distribution",fontweight='bold',fontsize=15)
plt.tick_params(labelsize=40)
plt.pie(flight.Accidents,labels=labels,textprops={'fontsize': 13});
plt.savefig('complaints.png', dpi=300)


# Refining The dataset:

# In[18]:


df.isnull().sum().sum()


# Hence,there are no null values!

# In[24]:


#Label encoding severity column and dropping airplane_id since it's irrelavnt:
le_Severity= LabelEncoder()
all_Severity = df.Severity
le_Severity.fit(all_Severity)

df['Severity'] = le_Severity.transform(df['Severity'])
df.drop(labels=['Accident_ID'],inplace=True,axis=1)


# In[25]:


df.head(3)


# In[27]:


#Dividing dataset into test and train sets:
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], 
                                                    df.iloc[:, -1], 
                                                    test_size = 0.3, 
                                                    random_state = 42)


# In[28]:


standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train = standardScaler.transform(X_train)
X_test = standardScaler.transform(X_test)


# In[29]:


linearRegression = LinearRegression()
linearRegression.fit(X_train, y_train)
y_pred = linearRegression.predict(X_test)
r2_score(y_test, y_pred)


# In[34]:


rf = RandomForestRegressor(n_estimators = 100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
r2_score(y_test, y_pred)

