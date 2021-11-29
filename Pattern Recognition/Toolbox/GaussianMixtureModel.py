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

import math
import numpy
import numpy.matlib

from KMeansClustering import KMeansClustering

"""
In this exercise, you will implement a classifier that uses Gaussian Mixture Models for classification.

1) Implement a Gaussian Mixture Model in GaussianMixtureModel.py.
   Use the initialization provided by KMeansClustering.py (reference implementation available for download below)

2) Implement a GMM-classifier in GMMClassifier.py .
   This classifier should estimate one GMM for each class-conditional distribution.
   Use your implementation in GaussianMixtureModel.py to estimate the GMM for each class.

3) Submit GaussianMixtureModel.py and GMMClassifier.py
"""


class GaussianMixtureModel(object):
    def __init__(self, numComponents, maxIterations=500):
        return None

    def fit(self, X):
        return None

    def getComponents(self, X):
        return None

    def getProbabilityScores(self, X):
        return None

    def evaluateGaussian(self, X, prior, mean, cov):
        return None

    def covDistance(self, covs1, covs2):
        return None

"""
Since this is classification task you might adapt the test case from exercise 1 or exercise2
"""
