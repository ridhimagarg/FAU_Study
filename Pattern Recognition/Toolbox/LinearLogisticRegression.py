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
Create a Logistic Regression classifier.
Implement your solution in LinearLogisticRegression.py.
Use an affine decision function (i.e.  a decision function that is linear with respect to the features with a constant appended to the feature vector).

This classifier only supports two-class problems.

Remember to encode the classes as 0/1.

Iteratively estimate the parameters of the decision function using the Newton method described in the lecture.
"""

import numpy as np



class LinearLogisticRegression(object):
    def __init__(self, learningRate=0.5, maxIterations=100):
        return None

    def fit(self, X, y):
        """
        Be careful when using the GUI classes can have different class numbers from 0/1.
        You need to map the to class number somehow to 0 and 1.
        """
        return None

    def gFunc(self, X, theta):
        return None

    def predict(self, X):
        return None



def test_linear_logistic_regression():
    """
    Use this to test your code before using the GUI.

    When it's working you can try your algorithm in the GUI

    1. start Toolbox/PyClassificationToolbox.py
    2. select "Classification > LinearLogisticRegression..." in the menu
    3. See your predication as background color
    4. Load a different data set

    When your algorithm is slow, try smaller window size!
    """
    num_classes = 2
    num_neighbors = 2
    num_training_samples = 10
    num_inference_samples = 15
    noise_magnitude = 0.2
    sample_spread = 10

    classifier = LinearLogisticRegression(num_neighbors)

    cluster_centers = sample_spread * np.random.rand(num_classes, 2)

    # This is our true y
    sample_labels_training = np.random.randint(num_classes, size=(num_training_samples,))
    # These are the positions of the training samples
    sample_position_training = cluster_centers[sample_labels_training] + \
        noise_magnitude * np.random.randn(num_training_samples, 2)

    # During training we know the labels
    classifier.fit(sample_position_training, sample_labels_training)

    # This is our true y
    sample_labels_inference = np.random.randint(num_classes, size=(num_inference_samples,))
    # These are the positions of the training samples
    sample_position_inference = cluster_centers[sample_labels_inference] + \
        noise_magnitude * np.random.randn(num_inference_samples, 2)

    # During inference we test what we have learned
    prediction = classifier.predict(sample_position_inference)

    correct = [p == t for p, t in zip(prediction, sample_labels_inference)]

    accuracy = sum(correct) / num_inference_samples
    print(f'{accuracy=}')


if __name__ == '__main__':
    test_linear_logistic_regression()
