#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


# Load the dataset

df = pd.read_csv('fish.csv')


# Data preprocessing

# In[3]:


print(df.isnull().sum())


# In[6]:


print(df.info())


# In[7]:


# Feature and target variable
X = df[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]  # Features
y = df['Species']  # Target variable


# In[8]:


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[9]:


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[10]:


# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 100, random_state = 42)
classifier.fit(X_train, y_train)


# In[11]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[12]:


# Making the Confusion Matrix
print(classification_report(y_test, y_pred))


# In[13]:


# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')


# In[16]:


import pickle

# Assuming 'classifier' is your trained model
model = classifier

# Save the model to disk
filename = 'fish_classifier.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved to {filename}")


# In[17]:


with open('sc.pkl', 'wb') as file:
    pickle.dump(sc, file)


# In[ ]:




