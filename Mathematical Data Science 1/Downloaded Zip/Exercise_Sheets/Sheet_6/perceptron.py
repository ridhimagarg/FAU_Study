import pandas as pd
import numpy as np


def heavyside(input):

    if input<0:
        return 0
    else:
        return 1


# def heavysideforand():

#     if input<=0:
#         return 0
#     else:
#         return 1

def not_logic(output):

    if output ==0:
        return 1
    if output ==1:
        return 0


def perceptron(x, theta):

    sum_ = theta[0] + theta[1]*x[0] + theta[2]*x[1]

    return heavyside(sum_)


for data_point in ([0,0], [0,1], [1,0], [1,1]):

    print(f"And for {data_point} is {perceptron(data_point, [-1.5, 1,1])}")## AND Implementation

    print(f"OR for {data_point} is {perceptron(data_point, [-1, 1,1])}")## OR Implementation

    print(f"NAND for {data_point} is {not_logic(perceptron(data_point, [-1.5, 1,1]))} ")

    print(f"NOR for {data_point} is {not_logic(perceptron(data_point, [-1, 1,1]))} ")

    print(f"XOR for {data_point} is {perceptron([perceptron(data_point, [-1.5, 1,1]), not_logic(perceptron(data_point, [-1, 1,1]))], [-1, 1,1])} ")

