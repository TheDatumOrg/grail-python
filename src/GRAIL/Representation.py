from GRAIL.GRAIL_core import GRAIL_rep, GRAIL_general
import numpy as np
import GRAIL.kernels as kernels
from GRAIL.exceptions import KernelNotFound

#Don't use. Does not work right now.
class GRAIL_general:
    def __init__(self, kernel = "SINK", d = 100, f = 0.99, r = 20, kernel_param_range = [*range(1,21)],
                 eigenvecMatrix = None, inVa = None,
                 kshape_initialization_method = "partition", dictionary_sampling = "random", **kwargs):
        if kernel == 'gak':
            self.kernel = kernels.gak
            self.kwargs = kwargs
        elif kernel == 'rbf':
            self.kernel = kernels.rbf
            self.kwargs = kwargs
        elif kernel == "kdtw":
            self.kernel = kernels.kdtw
            self.kwargs = kwargs
        else:
            raise KernelNotFound
        self.d = d
        self.f = f
        self.r = r
        self.kernel_param_range = kernel_param_range
        self.eigenvecMatrix = eigenvecMatrix
        self.inVa = inVa
        self.initialization_method = kshape_initialization_method
        self.dictionary_sampling = dictionary_sampling

    def get_representation(self, X):
        """
        Get the representation of matrix X
        :param X:
        :return:
        """
        if self.d > X.shape[0]:
            raise ValueError("The number of landmark series should be smaller than the number of time series.")
        Z_k, Zexact, self.best_gamma = GRAIL_general(X, self.kernel, self.d, self.f, self.r, self.kernel_param_range,
                                    self.eigenvecMatrix, self.inVa,
                                    self.initialization_method, self.dictionary_sampling, **self.kwargs)
        return Z_k

    def get_exact_representation(self, X):
        """
        Get the representation of matrix X
        :param X:
        :return:
        """
        if self.d > X.shape[0]:
            raise ValueError("The number of landmark series should be smaller than the number of time series.")
        Z_k, Zexact, self.best_gamma = GRAIL_general(X, self.kernel, self.d, self.f, self.r, self.GV, self.eigenvecMatrix, self.inVa,
                  self.initialization_method, self.dictionary_sampling, **self.kwargs)
        return Zexact

    def get_rep_train_test(self, TRAIN, TEST, exact = True):
        """
        Get Grail representation for TRAIN and TEST sets
        :param TRAIN:
        :param TEST:
        :return:
        """
        together = np.vstack((TRAIN, TEST))
        if exact:
              rep_together = self.get_exact_representation(together)
        else:
            rep_together = self.get_representation(together)
        rep_TRAIN = rep_together[0:TRAIN.shape[0], :]
        rep_TEST = rep_together[TRAIN.shape[0]:, :]
        return rep_TRAIN, rep_TEST





class GRAIL:

    def __init__(self, kernel = "SINK", d = 100, f = 0.99, r = 20, GV = [*range(1,21)],
                 fourier_coeff = -1, e = -1, eigenvecMatrix = None, inVa = None, gamma = None, sigma = None, initialization_method = "partition"):
        self.kernel = kernel
        self.d = d
        self.f = f
        self.r = r
        self.GV = GV
        self.fourier_coeff = fourier_coeff
        self.e = e
        self.eigenvecMatrix = eigenvecMatrix
        self.inVa = inVa
        self.initialization_method = initialization_method
        self.gamma = gamma
        self.sigma = sigma


    def get_representation(self, X):
        """
        Get the representation of matrix X
        :param X:
        :return:
        """
        if self.d > X.shape[0]:
            raise ValueError("The number of landmark series should be smaller than the number of time series.")
        if self.kernel == "SINK":
            Z_k, Zexact, self.best_gamma = GRAIL_rep(X, self.d, self.f, self.r, self.GV,
                                    self.fourier_coeff, self.e, self.eigenvecMatrix, self.inVa, self.gamma, self.initialization_method)
        elif self.kernel == "kdtw":
            pass
            #Z_k, Zexact, self.best_gamma = GRAIL_rep_kdtw(X, self.d, self.f, self.r, self.GV,
             #                                             self.sigma, self.eigenvecMatrix, self.inVa)
        return Z_k

    def get_exact_representation(self, X):
        """
        Get the representation of matrix X
        :param X:
        :return:
        """
        if self.d > X.shape[0]:
            raise ValueError("The number of landmark series should be smaller than the number of time series.")
        if self.kernel == "SINK":
            Z_k, Zexact, self.best_gamma = GRAIL_rep(X, self.d, self.f, self.r, self.GV, self.fourier_coeff, self.e, self.eigenvecMatrix, self.inVa,
                  self.gamma, self.initialization_method)
        elif self.kernel == "kdtw":
            #Z_k, Zexact, self.best_gamma = GRAIL_rep_kdtw(X, self.d, self.f, self.r, self.GV, self.sigma, self.eigenvecMatrix, self.inVa)
            pass
        return Zexact

    def get_rep_train_test(self, TRAIN, TEST, exact = True):
        """
        Get Grail representation for TRAIN and TEST sets
        :param TRAIN:
        :param TEST:
        :return:
        """
        together = np.vstack((TRAIN, TEST))
        if exact:
            rep_together = self.get_exact_representation(together)
        else:
            rep_together = self.get_representation(together)
        rep_TRAIN = rep_together[0:TRAIN.shape[0], :]
        rep_TEST = rep_together[TRAIN.shape[0]:, :]
        return rep_TRAIN, rep_TEST

