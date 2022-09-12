#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# # EDA

# In[4]:


#import dataset
data= pd.read_csv('bank-full.csv')
data.head()


# In[5]:


data.shape


# In[6]:


data[data.duplicated()]


# In[7]:


data.isna().sum()


# In[8]:


data.info()


# In[9]:


#Visualizing the data
categ = ['job','marital','education','default','housing','loan','contact','day','month','duration','pdays','poutcome','y']
for col in categ:
    plt.figure(figsize=(15,4))
    sns.barplot(data[col].value_counts().values,data[col].value_counts().index)
    plt.title(col)


# In[10]:


#label encoding non-numeric data
le = LabelEncoder()
data['y']= le.fit_transform(data['y'])
data['job']=le.fit_transform(data['job'])
data['marital']=le.fit_transform(data['marital'])
data['education']=le.fit_transform(data['education'])
data['default']=le.fit_transform(data['default'])
data['housing']=le.fit_transform(data['housing'])
data['contact']=le.fit_transform(data['contact'])
data['loan']=le.fit_transform(data['loan'])
data['poutcome']=le.fit_transform(data['poutcome'])


# In[11]:


data.head()


# In[12]:


data['month']=data['month'].replace(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
                                            ["1","2","3","4","5","6","7","8","9","10","11","12"])


# In[13]:


data.head()


# # Building model

# In[14]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# Dividing our data into input and output variables 
X= data.iloc[:,0:16]
Y= data.iloc[:,-1]


# In[15]:


#splitting the data into train and test 
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size = 0.2,random_state=0)


# In[16]:


xtrain.shape


# In[17]:


ytrain.shape


# In[18]:


xtest.shape


# In[19]:


ytest.shape


# In[20]:


#standardizing the data
STD =StandardScaler()
xtrain = STD.fit_transform(xtrain)
xtest =STD.fit_transform(xtest)


# In[26]:


#Logistic regression and fit the model on train dat
model = LogisticRegression()
model.fit(xtrain,ytrain)
pred = model.predict(xtest)


# ### Confusion matrix

# In[27]:


from sklearn.metrics import classification_report
print(classification_report(pred,ytest))


# In[28]:


cm = confusion_matrix(ytest,pred)
print(cm)


# In[34]:


#accuracy
print("Accuracy = ",(7803 + 228)/(7803  +228  +  835  +177)*100)


# ## ROC Curve

# In[30]:


from sklearn import metrics
metrics.plot_roc_curve(model, xtest, ytest) 

