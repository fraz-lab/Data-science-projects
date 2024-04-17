#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


from sklearn.model_selection import train_test_split
import pandas as pd
ds=pd.read_csv(r"C:\Users\Infra\Desktop\faraz khan - project3_Dataset.csv")
ds.head()


# In[8]:


x = ds.drop(["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"], axis=1)
y = ds["target"]






# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(X_train)
x_test_scaler = scaler.transform(X_test)

k = 3 # k=3 values has good precision and F1_Score
model = KNeighborsClassifier(n_neighbors=k)




# In[12]:


# Fit the model on the scaled training data and target values
model.fit(x_train_scaler, Y_train)


# In[13]:


# Predict the target values for the scaled test data
y_pred = model.predict(x_test_scaler)

accuracy = accuracy_score(Y_test, y_pred)
report = classification_report(Y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(report)

