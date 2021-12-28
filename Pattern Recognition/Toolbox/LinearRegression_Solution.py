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

import numpy
import numpy as np
import scipy.optimize


class LinearRegression(object):
    def __init__(self,
                 lossFunction='l2',
                 lossFunctionParam=0.001,
                 classification=False):
        self.__lossFunction = lossFunction
        self.__lossFunctionParam = lossFunctionParam
        self.__classification = classification

    def fit(self, X, y):
        self.__initialized = False

        if self.__classification:
            y = y.astype(numpy.int)
            classes = list(set(y))

            assert len(
                classes
            ) == 2, "This implementation of the Linear Regression classifier an handle only 2-class problems!"

            self.__mapping = {}
            self.__mapping[-1] = classes[0]
            self.__mapping[+1] = classes[1]
            y[y == classes[0]] = -1
            y[y != -1] = +1

        if len(y) < 2:
            return

        # special case: X is a 1-D numpy array
        # mx = numpy.mean(X)
        # my = numpy.mean(y)
        # mxy = numpy.mean(X * y)
        # a = (mxy - mx * my) / numpy.var(X)
        # b = my - a * mx
        # print("{0} * x + {1}".format(a, b))

        X = numpy.vstack((X.T, numpy.ones(len(X))))
        print(X)
        print(X.shape)
        self.params = numpy.dot(
            numpy.dot(numpy.linalg.inv(numpy.dot(X, X.T)), X), y)
        print(self.params.shape)
        test1 = numpy.linalg.pinv(X)
        test2 = numpy.linalg.pinv(X.T)
        params = numpy.dot(test2,y)

        if self.__lossFunction == 'huber':
            a = self.__lossFunctionParam

            res = scipy.optimize.minimize(
                lambda x: self.huber_objfunc(X, y, x, a),
                self.params,
                method='BFGS',
                jac=lambda x: self.huber_objfunc_derivative(X, y, x, a))
            # print(res)
            # print(res.x)
            self.params = res.x

        self.__initialized = True

    def huber_objfunc(self, X, y, params, a):
        r = y - numpy.dot(params, X)
        return numpy.sum(self.huber(r, a))

    def huber_objfunc_derivative(self, X, y, params, a):
        r = y - numpy.dot(params, X)
        return -numpy.dot(X, self.huber_derivative(r, a))

    def huber(self, r, a):
        huber = numpy.zeros(len(r))
        huber[abs(r) <= a] = r[abs(r) <= a]**2
        huber[abs(r) > a] = a * (2 * abs(r[abs(r) > a]) - a)
        return huber

    def huber_derivative(self, r, a):
        derivative = numpy.zeros(len(r))
        derivative[abs(r) <= a] = 2 * r[abs(r) <= a]
        derivative[abs(r) > a] = 2 * a * numpy.sign(r[abs(r) > a])
        return derivative

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
        if not self.__initialized:
            return None

        # print(self.params)
        a = self.params[0:-1]
        a0 = self.params[-1]
        Z = numpy.sign(numpy.sum(X * a, 1) + a0)
        Z[Z > 0] = self.__mapping[+1]
        Z[Z <= 0] = self.__mapping[-1]
        return Z

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