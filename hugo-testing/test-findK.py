#!/usr/bin/env python

import sys
import os 
import math
import glob
import itertools

import numpy 
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors as rdm
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem import AllChem

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Load Data

df = pd.read_excel('MR1_map4-clean.xlsx') # xlsx is not ideal , is easier to convert here than in Excel

# Save the DataFrame to CSV format with comma as the delimiter
df.to_csv('MR1_map4-clean.csv', index=False)  # Set index=False to exclude row numbers in the CSV

df = pd.read_csv('MR1_map4-clean.csv')

df.head()
df.describe()

# Select only the fingerprint bits columns for clustering

map4_fp = df.iloc[:, 2:].values # np array with shape (98, 1024) 98 molecules x 1024 fp / molecule

# map4_fp.shape


# Apply PCA to reduce dimensionality to 2 components

# Standardization (Z-score scaling)
std_scaler = StandardScaler()
data_std = std_scaler.fit_transform(map4_fp)

# Perform PCA on the standardized data
pca = PCA(n_components=2)
pca_std = pca.fit_transform(data_std) # Coordinates for std data

# Perform PCA on normal data 

pca_not_std = pca.fit_transform(map4_fp) # Coordinates for not std data

df['PCA_1-not_std'] = pca_not_std[:, 0]
df['PCA_2-not_std'] = pca_not_std[:, 1]
df['PCA_1-std'] = pca_std[:, 0]
df['PCA_2-std'] = pca_std[:, 1]

columns = ['Name', 'PCA_1-not_std', 'PCA_2-not_std', 'PCA_1-std', 'PCA_2-std']
results_df = df[columns]

cost_no_std = []

cost_std = []
for i in range(1, 25):
    kmeans = KMeans(n_init = 100, n_clusters=i, max_iter=500)
    kmeans.fit(pca_not_std) 
    # label (cluster assigned) for each molecule giving the coordinates (PCA_1 and PCA_2)
    # Compute cluster centers and predict cluster index for each sample.
    cost_no_std.append(kmeans.inertia_)

    kmeans.fit(pca_std)
    cost_std.append(kmeans.inertia_)


# plot the cost against K values
plt.plot(range(1, 25), cost_std, color ='b', linewidth ='3')
plt.xlabel("Value of K std")
plt.ylabel("Squared Error (Cost)")
plt.show() # clear the plot

plt.plot(range(1, 25), cost_no_std, color ='b', linewidth ='3')
plt.xlabel("Value of K not std")
plt.ylabel("Squared Error (Cost)")
plt.show() # clear the plot
 
# the point of the elbow is the
# most optimal value for choosing k