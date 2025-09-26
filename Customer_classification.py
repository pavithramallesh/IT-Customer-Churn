#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('Customer-Churn.csv')


# In[3]:


data.head()


# In[4]:


data.size


# In[5]:


data.shape


# In[6]:


data.columns


# In[7]:


data.dtypes


# In[8]:


data.isnull().sum()


# In[9]:


data.duplicated().sum()


# In[10]:


data.dtypes


# In[11]:


data.describe()


# In[12]:


data


# In[13]:


data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')


# In[14]:


data.dtypes


# In[15]:


data.isnull().sum()


# In[16]:


data = data.dropna()


# In[17]:


data.isnull().sum()


# In[18]:


# Select categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

# Select numeric columns
numeric_columns = data.select_dtypes(include=['number']).columns.tolist()


# In[19]:


categorical_columns


# In[20]:


numeric_columns


# In[21]:


categorical_count = data['customerID'].nunique()


# In[22]:


data['Contract'].value_counts()


# In[23]:


numeric_columns = data.select_dtypes(exclude=['object'])


# In[24]:


numeric_columns.corr()


# ## Visualization

# In[25]:


plt.scatter(data['gender'], data['tenure'])


# In[26]:


sns.pairplot(data)


# In[27]:


sns.countplot(data=data, y ='Contract', hue='gender');


# ## Model Evaluation

# In[28]:


data.head()


# In[29]:


from sklearn.preprocessing import LabelEncoder




# Select categorical columns
categorical_columns = data.select_dtypes(include=['object'])

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Iterate through each categorical column and transform it
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])


# In[30]:


data.head()


# In[31]:


X= data.drop(['Churn','customerID'],axis=1)
y= data['Churn']


# In[32]:


y


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



# In[34]:


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[35]:


# Creating the decision tree classifier
clf = DecisionTreeClassifier()
# Training the decision tree classifier
clf.fit(X_train, y_train)


# In[36]:


y_test


# In[37]:


y_pred = clf.predict(X_test)


# In[38]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report

# Evaluating the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[39]:


y_train_pred = clf.predict(X_train)
# Displaying the predicted values
print("Predictions on Training Data:")
print(y_train_pred)


# In[40]:


# Predicting on the test data
y_test_pred = clf.predict(X_test)

# Displaying the predicted values
print("Predictions on Test Data:")
print(y_test_pred)


# In[41]:


cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
print("Confusion Matrix:")
print(cm)


# In[42]:


a = confusion_matrix(y_test, y_pred)
sns.heatmap(a, annot=True)


# In[43]:


s = classification_report(y_test, y_pred)
print(s)


# In[ ]:





# In[ ]:




