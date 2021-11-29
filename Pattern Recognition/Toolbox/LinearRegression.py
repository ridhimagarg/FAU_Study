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
        self.__initialized = False  #set to true when done
        return None

    def fit(self, X, y):
        return None

    def huber_objfunc(self, X, y, params, a):
        return None

    def huber_objfunc_derivative(self, X, y, params, a):
        return None

    def huber(self, r, a):
        return None

    def huber_derivative(self, r, a):
        return None

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

    # Can the regression class determine m and a?
    r = LinearRegression()
    r.fit(x0, x1_noisy)

    print(f'{r.params=} vs {m=} {a=}')

    assert np.allclose(r.params[0], m, atol=9e-2)
    assert np.allclose(r.params[1], a, atol=9e-2)

    

if __name__ == '__main__':
    main()
