#    Copyright 2016 Stefan Steidl
#    Friedrich-Alexander-Universität Erlangen-Nürnberg
#    Lehrstuhl für Informatik 5 (Mustererkennung)
#    Martensstraße 3, 91058 Erlangen, GERMANY
#    stefan.steidl@fau.de

#    This file is part of the Python Classification Toolbox.
#
#    The Python Classification Toolbox is free software:
#    you can redistribute it and/or modify it under the terms of the
#    GNU General Public License as published by the Free Software Foundation,
#    either version 3 of the License, or (at your option) any later version.
#
#    The Python Classification Toolbox is distributed in the hope that
#    it will be useful, but WITHOUT ANY WARRANTY; without even the implied
#    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#    See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with the Python Classification Toolbox.
#    If not, see <http://www.gnu.org/licenses/>.
"""
Implement Linear Regression in LinearRegression.py.
The parameter 'classification' can be ignored, as well as the 'predict' function.
The parameter lossFunction can either be 'l2' for linear regression or 'huber'
for regression using a Huber loss function.
In this case, lossFunctionParam is the parameter of the Huber function. The 'fit' function should
compute the affine regression from X onto y (i.e. you need to extend the parameter vector with a constant).

Hint: to compute the regression parameters for the Huber function,
it may be easier use scipy.optimize.minimize on the pre-defined functions
than to implement the full optimization yourself. For the L2-Loss,
the regression can be computed using the pseudo-inverse.
"""

import numpy
import scipy.optimize
import numpy as np


class LinearRegression(object):
    def __init__(self,
                 lossFunction='l2',
                 lossFunctionParam=0.001,
                 classification=False):

        self.a = lossFunctionParam
        self.__initialized = False  #set to true when done
        self.maxIterations = 10

        return None

    def fit(self, X, y):

        X = np.vstack((X.T, np.ones(len(X))))

        self.params = numpy.dot(
            numpy.dot(numpy.linalg.inv(numpy.dot(X, X.T)), X), y)


        while self.maxIterations:

            self.params -= 0.1 * self.huber_objfunc_derivative(X, y, self.params, self.a)

            print(self.params)

            self.maxIterations -= 1

        # self.params = []

        # intitial_params = np.zeros((X.shape[1],1))
        # self.params.append(intitial_params)

        # initial_bias = np.zeros(1)
        # self.params.append(initial_bias)

        # y_pred = (X @ self.params[0] +self.params[1]).reshape(y.shape[0],)

        # y_pred = np.array([self.huber(x, self.a) for x in y-y_pred])

        # y_pred = (self.params @ X)



    def huber_objfunc(self, X, y, params, a):

        r = y - numpy.dot(params, X)

        return r

    def huber_objfunc_derivative(self, X, y, params, a):

        # r = y - numpy.dot(params, X)

        r = self.huber_objfunc(X, y, params, a)

        return -numpy.dot(X, np.array([self.huber_derivative(r_, a) for r_ in r]))

        # common_term = y- (X @ params[0] + params[1]) 

        # print(np.array([self.huber_derivative(r, a) for r in y-y_pred] * (y-y_pred)).reshape(y_pred.shape[0], 1).T @ X)

        # dW = np.array([self.huber_derivative(r, a) for r in y-y_pred] * (y-y_pred)).reshape(y_pred.shape[0], 1).T @ X

        # dB = np.sum(np.array([self.huber_derivative(r, a) for r in y-y_pred] * (y-y_pred)).reshape(y_pred.shape[0], 1).T * (-1))

        # self.params[0] -= 0.1* dW
        # self.params[1] -= 0.1* dB

        # print(self.params[0])
        # print(self.params[1])

        # return None

    def huber(self, r, a):

        if abs(r) <= a:
            return r*r

        else:
            return a*(2*abs(r) - a)


    def huber_derivative(self, r, a):
        
        if abs(r) <= a:
            return 2*r
        elif r>a:
            return 2*a
        else:
            return -2*a


    def paint(self, qp, featurespace):
        if self.__initialized:
            x_min, y_min, x_max, y_max = featurespace.coordinateSystem.getLimits(
            )
            y1 = self.params[0] * x_min + self.params[1]
            x1, y1 = featurespace.coordinateSystem.world2screen(x_min, y1)
            y2 = self.params[0] * x_max + self.params[1]
            x2, y2 = featurespace.coordinateSystem.world2screen(x_max, y2)
            qp.drawLine(x1, y1, x2, y2)

    def predict(self, X):
        return None


def main():

    m = 0.3
    a = 2
    noise = 0.1
    spread = 10
    num_samples = 40
    
    x0 = spread * np.random.rand(num_samples)
    x1 = m * x0 + a
    x1_noisy = x1 + noise * np.random.randn(num_samples)

    # print(x0.shape)
    # print(x1.shape)

    # print(x0.shape[0])

    # Can the regression class determine m and a?
    r = LinearRegression()
    r.fit(x0, x1_noisy)

    print(f'{r.params=} vs {m=} {a=}')

    assert np.allclose(r.params[0], m, atol=9e-2)
    assert np.allclose(r.params[1], a, atol=9e-2)

    

if __name__ == '__main__':
    main()
