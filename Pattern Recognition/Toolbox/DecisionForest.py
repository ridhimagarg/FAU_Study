import numpy
from DecisionTree import DecisionTree


class DecisionForest(object):
    def __init__(self, numTrees, depth, numSplit):
        self.__numTrees = numTrees
        self.__depth = depth
        self.__numSplit = numSplit

    def fit(self, X, y):

        self.__treeList = []
        for i in range(self.__numTrees):
            tree = DecisionTree(self.__depth, self.__numSplit)
            tree.fit(X, y)
            self.__treeList.append(tree)

    def predict(self, X):

        Z = numpy.zeros(len(X))
        for i in range(len(X)):
            Z[i] = self.predictSingle(X[i])

        return Z

    def predictSingle(self, X):
        forestPosteriors = [0 for i in range(10)]
        for i in range(self.__numTrees):
            treePosteriors = self.__treeList[i].predictSingle(X)
            for j in range(10):
                forestPosteriors[j] += treePosteriors[j] / self.__numTrees

        return numpy.int64(forestPosteriors.index(max(forestPosteriors)))
