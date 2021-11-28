import numpy as np
from numpy.lib.function_base import cov
import pandas as pd
import matplotlib.pyplot as plt
# from numpy import lingalg as LA
from sklearn import datasets
from scipy import linalg
from mpl_toolkits import mplot3d   
from sklearn.cluster import KMeans 

np.set_printoptions(precision=4)

iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y = iris.target

print(X)
print(y)

# data = pd.read_csv('../Sheet_1/faithful.csv')

# print(data.to_numpy())

# X = X.to_numpy()

print(np.mean(X, axis=0))

mean_ = np.mean(X, axis=0)

data_mean_sub = X - mean_
print(data_mean_sub)

cov_ = [np.dot(e, np.transpose(e)) for e in data_mean_sub]

print("N is", X.shape[0])

cov_ = np.dot(np.transpose(data_mean_sub), data_mean_sub)/X.shape[0]

# print(np.cov(np.transpose(data_mean_sub)))

print(cov_)
print(cov_.shape)

print(linalg.eig(cov_))

k= 3
# k =2 


## It will return a shape of (No. of features(4 in this case), k(no. of features for new data after PCA))
principal_components = linalg.eig(cov_)[1][0:k,:].T 


## Now the shape of data is (N, k(new no. of components))
## Findinf the new data space.
print(np.dot(data_mean_sub, principal_components)) 

reduced_data = np.dot(data_mean_sub, principal_components)

print(reduced_data)


fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(reduced_data[:,0], reduced_data[:,1], reduced_data[:,2], c=reduced_data[:,2], cmap='Greens')

plt.show()

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

print(kmeans.labels_)

print(np.unique(kmeans.labels_, return_counts=True))

kmeans = KMeans(n_clusters=3, random_state=0).fit(reduced_data)

print(kmeans.labels_)

print(np.unique(kmeans.labels_, return_counts=True))

# print(np.dot(data_mean_sub, linalg.eig(cov_)[1][0:2,:].T))
# print(len(cov_))

# print(data.shape)

# print(LA.eig())   