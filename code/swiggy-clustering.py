#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import prince


# In[2]:


plots_path = './plots/'


# In[3]:


df = pd.read_csv('swiggy-preprocessed.csv', index_col=[0])


# In[4]:


df.head()


# In[5]:


cat_vars = ['sub_area', 'area', 'cuisine1', 'cuisine2', 'rating_count', 'city']
num_vars = ['rating', 'cost']

df = df[cat_vars + num_vars]


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


df


# In[10]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# your data
X = df.values

# fit PCA
pca = PCA().fit(X)

# scree plot
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.savefig(plots_path + 'number-of-comp.png', bbox_inches='tight')
plt.show()


# In[11]:


mca = prince.MCA(n_components=2)
mca.fit(df)


# In[12]:


def elbow_kmeans(df):
    # Determine optimal number of clusters using WCSS
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init='auto')
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig(plots_path + 'elbow-method.png', bbox_inches='tight')
    plt.show()


# In[13]:


reduced_df = df[['sub_area', 'area' , 'cuisine1' , 'rating', 'cost']]


# In[14]:


elbow_kmeans(reduced_df)


# In[15]:


mca.__dict__


# In[16]:


# Get the row and column coordinates
row_coords = mca.row_coordinates(df)
col_coords = mca.column_coordinates(df)


# In[17]:


# Plot the row and column coordinates
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x=row_coords[0], y=row_coords[1], ax=ax)
sns.scatterplot(x=col_coords[0], y=col_coords[1], ax=ax)
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.legend(['Rows', 'Columns'])
ax.set_title('MCA Coordinates')
plt.savefig(plots_path + 'mca-cordinates.png', bbox_inches='tight')
plt.show()


# In[18]:


fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(mca.column_contributions_.T, cmap='coolwarm', ax=ax, annot=True, fmt='.1f')
ax.set_xlabel('Dimension')
ax.set_ylabel('Category')
ax.set_xticklabels(df.columns)
ax.set_title('Contribution of Categories to Dimensions')
plt.savefig(plots_path + 'contribution-of-cat.png', bbox_inches='tight')
plt.show()


# In[19]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, n_init='auto')
kmeans.fit(row_coords)
df['cluster'] = kmeans.labels_


# In[20]:


fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x=row_coords[0], y=row_coords[1], hue=df['cluster'], ax=ax)
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_title('MCA Clusters')
plt.savefig(plots_path + 'mca-clusters.png', bbox_inches='tight')
plt.show()


# In[21]:


cluster_means = df.groupby('cluster').mean()
print(cluster_means)


# In[22]:


# Example: Boxplot of rating by cluster
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(x='cluster', y='rating', data=df, ax=ax)
ax.set_xlabel('Cluster')
ax.set_ylabel('Rating')
ax.set_title('Boxplot of Rating by Cluster')
plt.savefig(plots_path + 'rating-by-cluster.png', bbox_inches='tight')
plt.show()


# In[23]:


from sklearn.metrics import silhouette_score
silhouette_score(row_coords, df['cluster'])


# The silhouette score measures the goodness of clustering based on the distance between points within the same cluster compared to the distance between points in different clusters. It ranges from -1 to 1, where a score close to 1 indicates that the clustering is good, while a score close to -1 indicates that the clustering is poor. Therefore, a higher silhouette score means better clustering.
# 
# In summary, silhouette score provide a measure of how well the data points are clustered. Higher scores indicate better clustering.
