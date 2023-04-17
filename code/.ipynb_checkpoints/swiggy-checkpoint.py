#!/usr/bin/env python
# coding: utf-8

# # CS 6220 Final Project
# 
# # Analyzing Swiggy Restaurant Catalog Data
# 
# ## Group: Amber
# ### Members: Sudhanva Narayana, Shreyas Prasad, Matthew Chan

# ## Table of Contents
# 
# * [Quick Inspection of the Data](#quick-inspection-of-the-data)
# * [Important Features that could be useful](#important-features-that-could-be-useful)

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


plots_path = './plots/'


# In[3]:


df = pd.read_csv('swiggy.csv')


# ## Quick Inspection of the Data

# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.columns


# ## Important Features that could be useful
# 
# - **city**: Location can be a major aspect of how restaurants and their properties can change
# - **rating**: Could be a major factor in deciding the quality of the restaurant
# - **rating_count**: Could be a major factor in deciding the authenticity of the ratings
# - **cost**: Important feature for deciding how to price the restaurants
# - **cuisine**: Could be a major factor in categorizing the restaurants based on trends and type of food

# In[9]:


def preprocess_city(entry):
    # Split the entry by comma separator
    values = entry.split(',')
    
    # Check number of values and modify as necessary
    if len(values) == 1:
        # Prepend the same value twice
        values = [values[0], values[0], values[0]]
    elif len(values) == 2:
        # Prepend the first value to the list
        values = [values[0]] + values
    
    # Join the values back into a comma separated string
    return ','.join(values)

def preprocess_cuisine(entry):
    values = entry.split(',')

    if len(values) == 1:
        values = [values[0]] + values
    
        # Join the values back into a comma separated string
        return ','.join(values)
    return ','.join(values)


# In[10]:


df['city'].value_counts()


# In[11]:


df['city'][df['city'].isna()]


# ### Imputing data

# In[12]:


df['city'] = df['city'].apply(preprocess_city)


# In[13]:


df['city'].value_counts()


# #### Spitting the city column into sub_area, area and city

# In[14]:


df[['sub_area', 'area', 'city']] = df['city'].str.split(',', expand=True)


# In[15]:


df


# ### Quick look after imputation

# In[16]:


df[['sub_area','area','city']].value_counts()


# ### Visualising Ratings

# In[17]:


# Define a colormap
cmap = sns.color_palette('tab10', len(df['rating_count'].value_counts()))

# Create a bar plot of the rating counts
ax = df['rating_count'].value_counts().plot(kind='bar', color=cmap, figsize=(12, 8))

# Set the title of the plot
ax.set_title('Distribution of Ratings')

# Set the label for the x-axis
ax.set_xlabel('Number of Ratings')

# Set the label for the y-axis
ax.set_ylabel('Frequency')

# Add a legend to the plot
ax.legend(['Count'], loc='upper right')

# Display the values on top of the bars
for i in ax.containers:
    ax.bar_label(i, label_type='edge', fontsize=10, padding=8)

# Display the plot
plt.savefig(plots_path + 'distribution-of-ratings.png', bbox_inches='tight')
plt.show()


# In[18]:


df['cost'] = df['cost'].str.slice(start=1).astype('float')


# In[19]:


def filter_outliers(df, column_name):
    mean = df[column_name].mean()
    std = df[column_name].std()
    filtered = df[column_name][(df[column_name] > mean - 2 * std) & (df[column_name] < mean + 2 * std)]
    return filtered


# ### Visualising Cost

# In[20]:


df['cost'] = filter_outliers(df, 'cost')


# In[21]:


df['cost'] = df['cost'].fillna(df['cost'].median())


# In[22]:


# Set up the histogram plot
fig, ax = plt.subplots(figsize=(12, 8))
n, bins, patches = ax.hist(df['cost'], bins=10)

# Customize plot details
ax.set_title('Cost Distribution')
ax.set_xlabel('Cost')
ax.set_ylabel('Frequency')
ax.set_xticks(bins)
ax.set_xticklabels(['${:.0f}'.format(b) for b in bins])
ax.legend(['Cost'])

# Add value labels to the top of each bar
for i in range(len(patches)):
    x = patches[i].get_x() + patches[i].get_width() / 2
    y = patches[i].get_height()
    ax.annotate('{:.0f}'.format(y), (x, y), xytext=(0, 5), 
                textcoords='offset points', ha='center', va='bottom')

# Display the plot
plt.savefig(plots_path + 'cost-distribution.png', bbox_inches='tight')
plt.show()


# ### Analsing Cuisime

# In[23]:


df['cuisine']


# In[24]:


df['cuisine'].value_counts()


# In[25]:


df = df.dropna(subset=['cuisine']).copy()


# In[26]:


df.loc[:, 'cuisine'] = df['cuisine'].apply(preprocess_cuisine)


# In[27]:


df['cuisine']


# In[28]:


cuisine_counts = df['cuisine'].str.split(',', expand=True).stack().value_counts()

# Set up the bar plot
fig, ax = plt.subplots(figsize=(30,8))
cuisine_counts.plot(kind='bar')

# Customize plot details
ax.set_title('Cuisine Counts')
ax.set_xlabel('Cuisine')
ax.set_ylabel('Count')
ax.legend(['Count'])

# Display the plot
plt.savefig(plots_path + 'cusine_counts.png', bbox_inches='tight')
plt.show()


# In[29]:


not_cuisines = [
    '8:15 To 11:30 Pm',
    'Tex-Mex',
    'SVANidhi Street Food Vendor',
    'Popular Brand Store',    'Bhutanese',
    'Default',
    'Special Discount from (Hotel Swagath)',
    'Bakery products',
    'Free Delivery ! Limited Stocks!',
    'Attractive Combos Available',
    'BEVERAGE',
    'Biryani - Shivaji Military Hotel',
    'Use Code JUMBO30 to avail',
    'Code valid on bill over Rs.99',
    'Meat',
    'Navratri Special',
    'Discount offer from Garden Cafe Express Kankurgachi',
    'Use code XPRESS121 to avail.',
    'Bowl Company',
    'MAX 2 Combos per Order!',
    'Grocery products'
]


# In[30]:


# Set a minimum threshold frequency
min_frequency = 0.01

# Count the frequency of each cuisine
cuisine_counts = df['cuisine'].str.split(',', expand=True).stack().value_counts()

# Filter out infrequent cuisines
frequent_cuisines = cuisine_counts[cuisine_counts >= min_frequency * len(df)]


# In[31]:


frequent_cuisines


# In[32]:


df = df[df['cuisine'].str.split(',', expand=True).isin(frequent_cuisines.index).any(axis=1)]


# In[33]:


df = df[~df['cuisine'].str.split(',', expand=True).isin(not_cuisines).any(axis=1)]


# In[34]:


cuisine_counts = df['cuisine'].str.split(',', expand=True).stack().value_counts()

# Set up the bar plot
fig, ax = plt.subplots(figsize=(30,8))
cuisine_counts.plot(kind='bar')

# Customize plot details
ax.set_title('Cuisine Counts')
ax.set_xlabel('Cuisine')
ax.set_ylabel('Count')
ax.legend(['Count'])

# Display the plot
plt.savefig(plots_path + 'cusine-counts-2.png', bbox_inches='tight')
plt.show()


# In[35]:


df[['cuisine1', 'cuisine2']] = df['cuisine'].str.split(',', expand=True)
df.drop(columns=['cuisine'], inplace=True)


# In[36]:


df.head()


# In[37]:


df['rating'] = df['rating'].replace('--', '0')


# In[38]:


df['rating'] = df['rating'].astype('float')


# In[39]:


# Set up the histogram plot
fig, ax = plt.subplots(figsize=(12, 8))
n, bins, patches = ax.hist(df['rating'], bins=10, color='green')

# Customize plot details
ax.set_title('Ratings Distribution')
ax.set_xlabel('Ratings')
ax.set_ylabel('Frequency')
ax.set_xticks(bins)
ax.legend(['Rating'])

# Add value labels to the top of each bar
for i in range(len(patches)):
    x = patches[i].get_x() + patches[i].get_width() / 2
    y = patches[i].get_height()
    ax.annotate('{:.0f}'.format(y), (x, y), xytext=(0, 5), 
                textcoords='offset points', ha='center', va='bottom')

# Display the plot
plt.savefig(plots_path + 'ratings-distribution.png', bbox_inches='tight')
plt.show()


# In[40]:


# Set up the histogram plot
fig, ax = plt.subplots(figsize=(12, 8))
n, bins, patches = ax.hist(df['rating'][df['rating'] != 0], bins=10, color='green')

# Customize plot details
ax.set_title('Ratings Distribution')
ax.set_xlabel('Ratings')
ax.set_ylabel('Frequency')
ax.set_xticks(bins)
ax.legend(['Rating'])

# Add value labels to the top of each bar
for i in range(len(patches)):
    x = patches[i].get_x() + patches[i].get_width() / 2
    y = patches[i].get_height()
    ax.annotate('{:.0f}'.format(y), (x, y), xytext=(0, 5), 
                textcoords='offset points', ha='center', va='bottom')

# Display the plot
plt.savefig(plots_path + 'ratings-distribution-2.png', bbox_inches='tight')
plt.show()


# In[41]:


df.head()


# In[42]:


df.to_csv('swiggy-preprocessed.csv', index=False)

