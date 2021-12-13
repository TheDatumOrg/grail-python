import numpy as np

#Gaussian Kernel
#tested
def DM2KM(DM):
    '''

    :param DM: nxn distance matrix of n time series
    :return: Kernel matrix
    '''

    n = DM.shape[0]
    sigma = np.mean(DM)
    KM = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            KM[i,j] = np.exp((-DM[i,j] ** 2)/(2*(sigma**2)))

    return KM

