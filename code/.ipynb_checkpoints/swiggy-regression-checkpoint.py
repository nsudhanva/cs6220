#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error


# In[3]:


df = pd.read_csv('swiggy-preprocessed.csv')


# In[4]:


df.head()


# In[5]:


X = df.drop('cost', axis=1)
y = df['cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


cat_cols = ['rating_count', 'sub_area', 'area', 'cuisine1', 'cuisine2', 'city']
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[7]:


preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_transformer, cat_cols)
])


# In[8]:


dt_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('regressor', DecisionTreeRegressor())
])

rf_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('regressor', RandomForestRegressor())
])


# In[9]:


dt_params = {
    'regressor__max_depth': [3, 5, 7],
    'regressor__min_samples_split': [2, 3, 4]
}

rf_params = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [5, 7],
    'regressor__min_samples_split': [2, 3],
    'regressor__max_features': ['sqrt', 'log2']
}


# In[10]:


dt_grid = GridSearchCV(dt_pipe, dt_params, cv=5)
rf_grid = GridSearchCV(rf_pipe, rf_params, cv=5)


# In[11]:


dt_grid.fit(X_train, y_train)
rf_grid.fit(X_train, y_train)


# In[12]:


dt_pred = dt_grid.predict(X_test)
rf_pred = rf_grid.predict(X_test)


# In[13]:


dt_r2 = r2_score(y_test, dt_pred)
dt_rmse = mean_squared_error(y_test, dt_pred, squared=False)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)

print(f'Decision Tree R2 score: {dt_r2:.4f}')
print(f'Decision Tree RMSE: {dt_rmse:.2f}')
print(f'Random Forest R2 score: {rf_r2:.4f}')
print(f'Random Forest RMSE: {rf_rmse:.2f}')

