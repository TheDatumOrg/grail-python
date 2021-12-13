from GRAIL.SINK import SINK
import numpy as np
import time
from GRAIL.FrequentDirections import FrequentDirections
from GRAIL.kshape import matlab_kshape, kshape_with_centroid_initialize
import GRAIL.exceptions as exceptions

def approx_gte(x, y):
    return np.logical_or((x > y), np.isclose(x, y))


def GRAIL_general(X, kernel, d, f, r, kernel_param_range, eigenvecMatrix = None,
              inVa = None, initialization_method = "partition", dictionary_sampling = "kshape", **kwargs):
    gamma = None
    n = X.shape[0]

    #Dictionary Sampling
    if dictionary_sampling == "kshape":
        if initialization_method == "partition":
            [mem, Dictionary] = matlab_kshape(X,d)
        elif initialization_method == "centroid_uniform":
            [mem, Dictionary] = kshape_with_centroid_initialize(X, d, is_pp=False)
        elif initialization_method == "k-shape++":
            [mem, Dictionary] = kshape_with_centroid_initialize(X, d, is_pp=True)
        else:
            raise exceptions.InitializationMethodNotFound

    elif dictionary_sampling == "random":
        dict_indices = np.random.choice(n, d, replace = False)
        Dictionary = X[dict_indices, :]

    else:
        raise exceptions.SamplingMethodNotFound


    # Tune the parameter of the kernel
    if kwargs['gamma_val'] == None and kernel == SINK:
        [score, kwargs['gamma_val']] = hyperparameter_select(Dictionary, kernel_param_range, r, kernel)
        gamma = kwargs['gamma_val']

    E = np.zeros((n, d))

    for i in range(n):
        for j in range(d):
            E[i, j] = kernel(X[i, :], Dictionary[j, :], **kwargs)

    if eigenvecMatrix == None and inVa == None:
        W = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                W[i, j] = kernel(Dictionary[i, :], Dictionary[j, :], **kwargs)

        [eigenvalvector, eigenvecMatrix] = np.linalg.eigh(W)
        inVa = np.diag(np.power(eigenvalvector, -0.5))

    Zexact = E @ eigenvecMatrix @ inVa
    Zexact = CheckNaNInfComplex(Zexact)
    Zexact = np.real(Zexact)

    BSketch = fd(Zexact, int(np.ceil(0.5 * d)))

    # eigh returns sorted eigenvalues in ascending order. We reverse this.
    [eigvalues, Q] = np.linalg.eigh(np.matrix.transpose(BSketch) @ BSketch)
    eigvalues = np.real(eigvalues)
    Q = np.real(Q)
    eigvalues = np.flip(eigvalues)
    Q = np.flip(Q)

    VarExplainedCumSum = np.divide(np.cumsum(eigvalues), np.sum(eigvalues))
    k = np.argwhere(approx_gte(VarExplainedCumSum, f))[0, 0] + 1
    Z_k = CheckNaNInfComplex(Zexact @ Q[0:d, 0:k])
    return Z_k, Zexact, gamma


def GRAIL_rep(X, d, f, r, GV, fourier_coeff = -1, e = -1, eigenvecMatrix = None, inVa = None, gamma = None, initialization_method = "partition"):
    """
    :param X: nxm matrix that contains n time series
    :param d: number of landmark series to extract from kshape
    :param f: scalar to tune the dimensionality k of Z_k
    :param r: parameter for tuning gamma, taken as 20 in the paper
    :param GV: vector of gammas to select the best gamma
    :param fourier_coeff: number of fourier coeffs to keep
    :param e: preserved energy in Fourier domain
    :return: Z_k nxk matrix of low dimensional reduced representation
    """

    n = X.shape[0]
    if initialization_method == "partition":
        [mem, Dictionary] = matlab_kshape(X,d)
    elif initialization_method == "centroid_uniform":
        [mem, Dictionary] = kshape_with_centroid_initialize(X, d, is_pp=False)
    elif initialization_method == "k-shape++":
        [mem, Dictionary] = kshape_with_centroid_initialize(X, d, is_pp=True)
    else:
        raise exceptions.InitializationMethodNotFound

    if gamma == None:
        [score, gamma] = gamma_select(Dictionary, GV, r)

    E = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            E[i, j] = SINK(X[i, :], Dictionary[j, :], gamma,fourier_coeff, e)

    if eigenvecMatrix == None and inVa == None:
        W = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                W[i, j] = SINK(Dictionary[i, :], Dictionary[j, :], gamma,fourier_coeff, e)

        [eigenvalvector, eigenvecMatrix] = np.linalg.eigh(W)
        inVa = np.diag(np.power(eigenvalvector, -0.5))

    Zexact = E @ eigenvecMatrix @ inVa
    Zexact = CheckNaNInfComplex(Zexact)
    Zexact = np.real(Zexact)

    BSketch = fd(Zexact, int(np.ceil(0.5 * d)))

    # eigh returns sorted eigenvalues in ascending order. We reverse this.
    [eigvalues, Q] = np.linalg.eigh(np.matrix.transpose(BSketch) @ BSketch)
    eigvalues = np.real(eigvalues)
    Q = np.real(Q)
    eigvalues = np.flip(eigvalues)
    Q = np.flip(Q)

    VarExplainedCumSum = np.divide(np.cumsum(eigvalues), np.sum(eigvalues))
    k = np.argwhere(approx_gte(VarExplainedCumSum, f))[0, 0] + 1
    Z_k = CheckNaNInfComplex(Zexact @ Q[0:d, 0:k])
    return Z_k, Zexact, gamma


def CheckNaNInfComplex(Z):
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isnan(Z[i, j]) or np.isinf(Z[i, j]) or (not np.isreal(Z[i, j])):
                Z[i, j] = 0

    return Z


#A function in the original MATLAB implementation. Not used here.
def repLearnKM(KM):
    t = time.time()
    [eigenvalvector, Q] = np.linalg.eigh(KM)
    eigenvalvector = np.flip(eigenvalvector)
    Q = np.flip(Q)

    VarExplainedCumSum = np.divide(np.cumsum(eigenvalvector), np.sum(eigenvalvector))

    DimFor99 = np.argwhere(VarExplainedCumSum >= 0.99)[0, 0] + 1
    DimFor98 = np.argwhere(VarExplainedCumSum >= 0.98)[0, 0] + 1
    DimFor97 = np.argwhere(VarExplainedCumSum >= 0.97)[0, 0] + 1
    DimFor95 = np.argwhere(VarExplainedCumSum >= 0.95)[0, 0] + 1
    DimFor90 = np.argwhere(VarExplainedCumSum >= 0.90)[0, 0] + 1
    DimFor85 = np.argwhere(VarExplainedCumSum >= 0.85)[0, 0] + 1
    DimFor80 = np.argwhere(VarExplainedCumSum >= 0.80)[0, 0] + 1

    RepLearnTime = time.time() - t

    Z99per = CheckNaNInfComplex(Q[:, 0: DimFor99] @ np.sqrt(np.diag(eigenvalvector[0: DimFor99])))
    Z98per = CheckNaNInfComplex(Q[:, 0: DimFor98] @ np.sqrt(np.diag(eigenvalvector[0: DimFor98])))
    Z97per = CheckNaNInfComplex(Q[:, 0: DimFor97] @ np.sqrt(np.diag(eigenvalvector[0: DimFor97])))
    Z95per = CheckNaNInfComplex(Q[:, 0: DimFor95] @ np.sqrt(np.diag(eigenvalvector[0: DimFor95])))
    Z90per = CheckNaNInfComplex(Q[:, 0: DimFor90] @ np.sqrt(np.diag(eigenvalvector[0: DimFor90])))
    Z85per = CheckNaNInfComplex(Q[:, 0: DimFor85] @ np.sqrt(np.diag(eigenvalvector[0: DimFor85])))
    Z80per = CheckNaNInfComplex(Q[:, 0: DimFor80] @ np.sqrt(np.diag(eigenvalvector[0: DimFor80])))

    Ztop20 = CheckNaNInfComplex(Q[:, 0: 20] @ np.sqrt(np.diag(eigenvalvector[0: 20])))
    Ztop10 = CheckNaNInfComplex(Q[:, 0: 10] @ np.sqrt(np.diag(eigenvalvector[0: 10])))
    Ztop5 = CheckNaNInfComplex(Q[:, 0: 5] @ np.sqrt(np.diag(eigenvalvector[0: 5])))
    return [Z99per, Z98per, Z97per, Z95per, Z90per, Z85per, Z80per, Ztop20, Ztop10, Ztop5, RepLearnTime]


def gamma_select(Dictionary, GV, r, k=-1):
    """
    Parameter Tuning function. Tunes the parameters for GRAIL
    This function does not work at the moment for gamma values that don't correspond
    to a range(n)
    :param Dictionary: Dictionary to summarize the dataset. Provided by KShape
    :param GV: A vector of Gamma values to choose from.
    :param k: Number of Fourier coeffs to keep when computing the SINK function.
    :param r: The number of top eigenvalues to be considered. This is 20 in the paper.
    :return: the tuned parameter gamma and its score
    """
    d = Dictionary.shape[0]
    GVar = np.zeros(len(GV) + 5)
    var_top_r = np.zeros(len(GV) + 5)
    score = np.zeros(len(GV) + 5)
    for gamma in GV:
        W = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                W[i, j] = SINK(Dictionary[i, :], Dictionary[j, :], gamma, k)
        GVar[gamma] = np.var(W)
        [eigenvalvector, eigenvecMatrix] = np.linalg.eigh(W)
        eigenvalvector = np.flip(eigenvalvector)
        eigenvecMatrix = np.flip(eigenvecMatrix)
        var_top_r[gamma] = np.sum(eigenvalvector[0:r]) / np.sum(eigenvalvector)

    score = GVar * var_top_r
    best_gamma = np.argmax(score)
    best_score = score[best_gamma]
    return [best_score, best_gamma]

def hyperparameter_select(Dictionary, GV, r, kernel):
    """
    Parameter Tuning function. Tunes the parameters for kernels
    :param Dictionary: Dictionary to summarize the dataset.
    :param GV: A vector of sigma values to choose from.
    :param r: The number of top eigenvalues to be considered. This is 20 in the paper.
    :return: the tuned parameter sigma and its score
    """
    d = Dictionary.shape[0]
    GVar = np.zeros(len(GV))
    var_top_r = np.zeros(len(GV))
    score = np.zeros(len(GV))
    for i in range(len(GV)):
        param = GV[i]
        W = np.zeros((d, d))
        for j in range(d):
            for k in range(d):
                W[j, k] = kernel(Dictionary[j, :], Dictionary[k, :], param)
        GVar[i] = np.var(W)
        [eigenvalvector, eigenvecMatrix] = np.linalg.eigh(W)
        eigenvalvector = np.flip(eigenvalvector)
        eigenvecMatrix = np.flip(eigenvecMatrix)
        var_top_r[i] = np.sum(eigenvalvector[0:r]) / np.sum(eigenvalvector)

    score = GVar * var_top_r
    best_param_index = np.argmax(score)
    best_param = GV[best_param_index]
    best_score = score[best_param_index]
    print("Selected parameter = ", best_param)
    return [best_score, best_param]


# Frequent directions helper function returns sketch matrix of size (ell x d)
def fd(A, ell):
    """
    Returns a sketch matrix of size ell x A.shape[1]
    :param A: Matrix to be sketched
    :param ell:
    :return:
    """
    d = A.shape[1]
    sketcher = FrequentDirections(d, ell)
    for i in range(A.shape[0]):
        sketcher.append(A[i, :])
    sketch = sketcher.get()
    return sketch


def choose_d(TRAIN, train_labels, TEST):
    return int(min(max(4*len(np.unique(train_labels)), np.ceil(0.4 * (TRAIN.shape[0] + TEST.shape[0])), 20), 100))

