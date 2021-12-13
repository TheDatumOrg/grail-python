import numpy as np
from tslearn.metrics import gak

def rbf(x, y, gamma):
    K = np.linalg.norm(x, y) ** 2
    K *= -gamma
    np.exp(K, K)  # exponentiate K in-place
    return K

def gak(s1, s2, sigma = 1.0):
    return gak(s1, s2, sigma)

# Python source code for the "Kernelized" Dynamic Time Warping similarity (as defined in the reference below).
# Author: Pierre-Francois Marteau
# Version: V1.0 du 13/09/2014,
# Licence: GPL
# ******************************************************************
# This software and description is free delivered "AS IS" with no
# guaranties for work at all. Its up to you testing it modify it as
# you like, but no help could be expected from me due to lag of time
# at the moment. I will answer short relevant questions and help as
# my time allow it. I have tested it played with it and found no
# problems in stability or malfunctions so far.
# Have fun.
# *****************************************************************
# Please cite as:
# @article{marteau:hal-00486916,
#   AUTHOR = {Marteau, Pierre-Francois and Gibet, Sylvie},
#   TITLE = {{On Recursive Edit Distance Kernels with Application to Time Series Classification}},
#   JOURNAL = {{IEEE Transactions on Neural Networks and Learning Systems}},
#   PAGES = {1-14},
#   YEAR = {2014},
#   MONTH = Jun,
#   KEYWORDS = {Elastic distance, Time warp kernel, Time warp inner product, Definiteness, Time series classification, SVM},
#   DOI = {10.1109/TNNLS.2014.2333876},
#   URL = {http://hal.inria.fr/hal-00486916}
# }
#
''''
# function kdtw(A, B, sigma, epsilon=1e-3)
# Dynamic programming implementation of KDTW kernel
# input A: first multivariate time series: array of array (nxd), n is the number of sample, d is the dimension of each sample
# intput B: second multivariate time series: array of array (nxd), n is the number of sample, d is the dimension of each sample
# input sigma: >0 used in the exponential local kernel 
# output similarity: similarity between A and B (the higher, the more similar)
'''


def kdtw(A, B, sigma=1, epsilon=1e-3):
    d = np.shape(A)[1]
    Z = [np.zeros(d)]
    A = np.concatenate((Z, A), axis=0)
    B = np.concatenate((Z, B), axis=0)
    [la, d] = np.shape(A)
    [lb, d] = np.shape(B)

    DP = np.zeros((la, lb))
    DP1 = np.zeros((la, lb));
    DP2 = np.zeros(max(la, lb));
    l = min(la, lb);
    DP2[1] = 1.0;
    for i in range(1, l):
        DP2[i] = Dlpr(A[i], B[i], sigma, epsilon);
    if la < lb:
        for i in range(la, lb):
            DP2[i] = Dlpr(A[la - 1], B[i], sigma, epsilon);
    elif lb < la:
        for i in range(lb, la):
            DP2[i] = Dlpr(A[i], B[lb - i], sigma, epsilon);

    DP[0, 0] = 1;
    DP1[0, 0] = 1;
    n = len(A);
    m = len(B);

    for i in range(1, n):
        DP[i, 1] = DP[i - 1, 1] * Dlpr(A[i], B[2], sigma, epsilon);
        DP1[i, 1] = DP1[i - 1, 1] * DP2[i];

    for j in range(1, m):
        DP[1, j] = DP[1, j - 1] * Dlpr(A[2], B[j], sigma, epsilon);
        DP1[1, j] = DP1[1, j - 1] * DP2[j];

    for i in range(1, n):
        for j in range(1, m):
            lcost = Dlpr(A[i], B[j], sigma, epsilon);
            DP[i, j] = (DP[i - 1, j] + DP[i, j - 1] + DP[i - 1, j - 1]) * lcost;
            if i == j:
                DP1[i, j] = DP1[i - 1, j - 1] * lcost + DP1[i - 1, j] * DP2[i] + DP1[i, j - 1] * DP2[j]
            else:
                DP1[i, j] = DP1[i - 1, j] * DP2[i] + DP1[i, j - 1] * DP2[j];
    DP = DP + DP1;
    return DP[n - 1, m - 1]


# local similarity between two samples
# a: 1d numpy array
# b: 1d numpy array
# input sigma: >0 used in the exponential local kernel
# input epsilon: 1 > epsilon > 0
# return the local matching similarity (probability)
def Dlpr(a, b, sigma=1, epsilon=1e-3):
    return (np.exp(-np.sum((a - b) ** 2) / sigma) + epsilon) / (3 * (1 + epsilon))

