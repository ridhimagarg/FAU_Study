

import numpy as np
from pandas.io.parsers import read_csv
import pandas as pd
import matplotlib.pyplot as plt


## Defining the kernels
def poly_kernel(a, d, x1, x2):

    return (x1 @ x2 + a)**d

def gaussian_kernel(sigma, x1, x2):

    # return np.exp(-(np.square(dp-mean))/2*sigma**2)
    return np.exp(-np.linalg.norm(x1-x2)**2/2*sigma**2)


def read_circle_data(filename):

    columns = ['X1', 'X2', 'Class']
    dataframe = pd.read_csv(filename, names=columns)

    print(dataframe.shape)
    

    data_numeric = dataframe.loc[:, ~dataframe.columns.str.contains('Class')]
    classes = dataframe.loc[:, dataframe.columns.str.contains("Class")]


    data_numeric = data_numeric.to_numpy()
    classes = classes.to_numpy()

    print(data_numeric.shape)
    print(classes.shape)

    return data_numeric, classes


def kernelPCA(data_numeric, kernel, k, N):
    # data numeric: some input data
    # kernel: a function handle; which kernel to use
    # k: how many components do you want
    # N: how much data to use to calculate the components

    M = data_numeric.shape[1] ## dimensions of data
    K = np.ndarray([N, N]) ## kernel matrix of shape no. of data points taken to calculate kernel matric
    # K = np.ndarray([M, M])
    oneK = np.ndarray([N, N])

    # print(np.mean(data_numeric, axis=0))

    # mean_ = np.mean(data_numeric, axis=0)

    for i in range(N):
        for j in range(N):
            K[i, j] = kernel(data_numeric[i], data_numeric[j])
            oneK[i, j] = 1./N


    ## This is the way to center data using kernel ## formula given in slide also ## No covariance matrix
    Ktilde = K- oneK @ K - K @ oneK + oneK @ K @ oneK

    ## Now calculating the eigen value
    w, v = np.linalg.eig(Ktilde)

    w = np.real(w)
    v = np.real(v)

    # print(np.flip(w[np.argsort(w)]))
    # print(np.flip(v[:,np.argsort(w)]))

    w = np.flip(w[np.argsort(w)], axis=0)
    v = np.flip(v[:,np.argsort(w)], axis=1)

    ## Choosing k elements.
    print(w[range(k)])
    print(v[:,range(k)])

    ## Normalizing the eigen vector
    for i in range(k):
        scale = np.sqrt(1./w[i]*N*v[:,i] @ v[:, i])
        v[:,i] = v[:,i] * scale

    ## Transform the data points.
    def transform_function(data_point):
        return np.array([sum(v[i, j] * kernel(data_numeric[i], data_point) for i in range(N)) for j in range(k)])

    return transform_function


data_numeric, classes = read_circle_data("Circledata.sec")

color_dict = {0: "red", 1: "blue"}
color_list = [color_dict[int(label)] for label in classes]

k=2

transform_function = kernelPCA(data_numeric, lambda x1, x2: gaussian_kernel(1., x1, x2), k, 10)
transformed_data = np.array( [transform_function(x_) for x_ in data_numeric] )

plt.scatter(data_numeric[:, 0], data_numeric[:, 1], color=color_list)
plt.show()

plt.scatter(transformed_data[:, 0], transformed_data[:, 1], color=color_list)
plt.show()

transform_function = kernelPCA(data_numeric, lambda x1, x2: poly_kernel(1., 2, x1, x2), k, 10)
transformed_data = np.array( [transform_function(x_) for x_ in data_numeric] )

print(transformed_data)

plt.scatter(transformed_data[:, 0], transformed_data[:, 1], color=color_list)
plt.show()
