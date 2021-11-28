

import numpy as np
from pandas.io.parsers import read_csv
import pandas as pd
import matplotlib.pyplot as plt


def gaussian_kernel(sigma, dp, mean):

    # return np.exp(-(np.square(dp-mean))/2*sigma**2)
    return np.exp(-np.linalg.norm(dp-mean)**2/2*sigma**2)


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

    M = data_numeric.shape[1]
    K = np.ndarray([N, N])
    oneK = np.ndarray([N, N])

    for i in range(N):
        for j in range(N):
            K[i, j] = kernel(data_numeric[i], data_numeric[j])
            oneK[i, j] = 1./N

    Ktilde = (K - oneK @ K - K @ oneK + oneK @ K @ oneK)

    print(Ktilde.shape)

    w, v = np.linalg.eig(Ktilde)

    print(w)
    print(v)
    print(v.shape)

    w = w[range(k)]
    v = v[:, range(k)]

    print(v.shape)
    print(data_numeric.shape)
    print(Ktilde.shape)

    projected_points = np.dot(Ktilde, v)
    # print(projected_points)

    return projected_points



    # for i in range(N):
        # kernel(data_numeric[i], data_point)

data_numeric, classes = read_circle_data("Circledata.sec")

color_dict = {0: "red", 1: "blue"}
color_list = [color_dict[int(label)] for label in classes[:10]]

k=2

projected_points = kernelPCA(data_numeric, lambda x1, x2: gaussian_kernel(1., x1, x2), k, 10)

print(projected_points)

plt.scatter(projected_points[:, 0], projected_points[:, 1], color=color_list)
plt.show()
