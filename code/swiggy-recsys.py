#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import prince

from sklearn.preprocessing import StandardScaler, LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors


# In[2]:


plots_path = './plots/'


# In[3]:


df = pd.read_csv('swiggy-preprocessed.csv', index_col=[0])
df_og = df.copy()


# In[4]:


df.head()


# In[5]:


cat_vars = ['sub_area', 'area', 'cuisine1', 'cuisine2', 'rating_count', 'city']
num_vars = ['rating', 'cost']


# In[6]:


# Calculate the mean rating for each combination of 'sub_area', 'area', 'city', 'cuisine1', 'cuisine2'
mean_ratings = df.groupby(['sub_area', 'area', 'city', 'cuisine1', 'cuisine2'])['rating'].mean()

# Define a function to replace 0 rating with the mean for its corresponding combination of 'sub_area', 'area', 'city', 'cuisine1', 'cuisine2'
def replace_rating(row):
    if row['rating'] == 0:
        return mean_ratings.loc[row['sub_area'], row['area'], row['city'], row['cuisine1'], row['cuisine2']]
    else:
        return row['rating']

# Apply the function to the 'rating' column
df['rating'] = df.apply(replace_rating, axis=1)


# In[7]:


le = LabelEncoder()


# In[8]:


# Encode the categorical variables using LabelEncoder
for col in cat_vars:
    df[col] = le.fit_transform(df[col])


# In[9]:


# import required libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# In[10]:


# Select the relevant columns from the dataset
df = df[['sub_area', 'area', 'cuisine1', 'cuisine2', 'rating_count', 'city', 'rating', 'cost']]

# Scale the 'rating' and 'cost' columns between 0 and 1
scaler = MinMaxScaler()
df[['rating', 'cost']] = scaler.fit_transform(df[['rating', 'cost']])


# In[11]:


# Encode the categorical variables using one-hot encoding
encoded_data = pd.get_dummies(df, columns=['sub_area', 'area', 'cuisine1', 'cuisine2', 'city', 'rating_count'])


# In[12]:


from sklearn.neighbors import NearestNeighbors

# Fit the NearestNeighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(encoded_data)


# In[13]:


def recommend_nn(id, model, data, k=5):
    # Find the index of the restaurant with the given id
    index = data.index.get_loc(id)

    # Get the indices and distances of the k-nearest neighbors
    distances, indices = model.kneighbors(data.iloc[index].values.reshape(1, -1), n_neighbors=k+1)

    # Remove the index of the queried restaurant from the indices list
    indices = indices.squeeze().tolist()[1:]

    # Return the ids of the k-nearest neighbors
    return [data.iloc[i].name for i in indices]


# In[14]:


le_dict = {}
for var in cat_vars:
    df[var] = le.fit_transform(df[var])
    le_dict[var] = le


# In[15]:


# Test the recommendation function using NearestNeighbors
restaurant_id = 531342  # Janta Sweet House
recommended_ids = recommend_nn(restaurant_id, model, encoded_data)
print(recommended_ids)


# In[16]:


# Create a new dataframe to store recommended restaurants
recommended_df = pd.DataFrame(recommended_ids, columns=['id'])
recommended_df = pd.merge(recommended_df, df, on='id')


# In[17]:


# Reverse transform categorical variables
for var in cat_vars:
    le = le_dict[var]
    recommended_df[var] = le.inverse_transform(recommended_df[var].astype(int))

# Display recommended restaurants
print("Recommended restaurants based on restaurant_id = ", restaurant_id)
for id in recommended_ids:
    print(df_og.loc[id][['name', 'city', 'rating', 'rating_count', 'cost', 'lic_no', 'link', 'address', 'sub_area', 'area', 'cuisine1', 'cuisine2']])
    print()


# In[18]:


# Plot the distribution of restaurant ratings
plt.hist(df['rating'], bins=10)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Restaurant Ratings')
plt.savefig(plots_path + 'distribution-of-rest-ratings.png', bbox_inches='tight')
plt.show()

# Plot the distribution of restaurant costs
plt.hist(df['cost'], bins=10)
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.title('Distribution of Restaurant Costs')
plt.savefig(plots_path + 'distribution-of-rest-costs.png', bbox_inches='tight')
plt.show()

