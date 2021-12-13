import numpy as np
import GRAIL.SINK as SINK
from scipy import stats
from scipy import linalg
import random
import math
import heapq


def zscore(x, axis=0, ddof=0):
    return np.nan_to_num(stats.zscore(x, axis=axis, ddof=ddof))


# somewhat tested
def sbd(x, y):
    '''
    Shape based distance
    :param x: z-normalized time series x
    :param y: z-normalized time series y
    :return: The distance, the shift, and the aligned sequence
    '''
    ncc_seq = SINK.NCC(x, y)
    index = np.argmax(ncc_seq)
    value = ncc_seq[index]

    dist = 1 - value
    shift = index - len(x) + 1  # make sure this is true
    if shift > 0:
        yshifted = np.concatenate((np.zeros(shift), y[0:-shift]))
    elif shift == 0:
        yshifted = y
    else:
        yshifted = np.concatenate((y[-shift:], np.zeros(-shift)))

    return [dist, shift, yshifted]


def kshape_centroid(X, mem, ref_seq, k):
    '''
    Computes the centroid for the kshape algorithm
    :param X: the matrix of time series
    :param mem: partition membership array
    :param ref_seq: the reference sequence time series are aligned against
    :param k: the partition number we want the centroid for
    :return: centroid of partition k
    '''
    partition = np.array([]).reshape(0, X.shape[1])
    for i in range(X.shape[0]):
        if mem[i] == k:
            partition = np.vstack((partition, X[i, :]))

    # return all zeros if partition is empty
    if partition.shape[0] == 0:
        return np.zeros((1, X.shape[1]))

    if sum(ref_seq) != 0:
        for i in range(partition.shape[0]):
            [dist, shift, vshifted] = sbd(ref_seq, partition[i, :])
            partition[i, :] = vshifted

    n = partition.shape[0]
    m = partition.shape[1]
    z_partition = zscore(partition, axis=1, ddof=1)

    S = np.transpose(z_partition) @ z_partition
    Q = np.identity(m) - (1 / m) * np.ones(m)
    M = np.transpose(Q) @ S @ Q
    eigval, centroid = linalg.eigh(M, subset_by_index=[m - 1, m - 1])
    centroid = centroid.transpose()

    d1 = ED(partition[0, :], centroid)
    d2 = ED(partition[0, :], -centroid)
    if d1 < d2:
        return centroid
    else:
        return -centroid


def ED(x, y):
    return np.sqrt(np.sum(np.power(x - y, 2)))


def matlab_kshape(A, k):
    '''
    shape based clustering algorithm
    This is the version where we randomly initialize the partitions
    :param X: mxn matrix containing time series that are z-normalized
    :param k: number of clusters
    :return: index is the length n array containing the index of the clusters to which
    the series are assigned. centroids is the kxm matrix containing the centroids of
    the clusters
    '''

    m = A.shape[0]
    mem = np.zeros(m)

    for i in range(m):
        mem[i] = random.randrange(k)
    cent = np.zeros((k, A.shape[1]))

    for iter in range(100):
        prev_mem = mem.copy()
        cluster_cnt = np.zeros(k)
        empty_cluster_cnt = 0
        D = math.inf * np.ones((m, k))

        for i in range(k):
            cent[i, :] = kshape_centroid(A, mem, cent[i, :], i)
            cent[i, :] = zscore(cent[i, :], ddof=1)

        for i in range(m):
            for j in range(k):
                dist = 1 - max(SINK.NCC(A[i, :], cent[j, :]))
                D[i, j] = dist

        for i in range(m):
            mem[i] = np.argmin(D[i, :])
            cluster_cnt[int(mem[i])] = cluster_cnt[int(mem[i])] + 1

        # check for empty clusters
        empty_cluster_list = []
        for cluster in range(k):
            if cluster_cnt[cluster] == 0:
                empty_cluster_cnt = empty_cluster_cnt + 1
                empty_cluster_list.append(cluster)

        # deal with empty clusters
        if empty_cluster_cnt != 0:
            min_dists = np.amin(D, axis=1)
            templist = np.array(heapq.nlargest(empty_cluster_cnt, enumerate(min_dists), key = lambda x: x[1]))
            distant_points = templist[:, 0]
            for i in range(empty_cluster_cnt):
                mem[int(distant_points[i])] = empty_cluster_list[i]


        if linalg.norm(prev_mem - mem) == 0:
            for i in range(k):
                cent[i, :] = kshape_centroid(A, mem, cent[i, :], i)
                cent[i, :] = zscore(cent[i, :], ddof=1)
            break

    return [mem, cent]


def kshape_with_centroid_initialize(X, k, is_pp = True):
    '''
    shape based clustering algorithm
    This is the version where we randomly initialize the centroids.
    :param X: nxm matrix containing time series that are z-normalized
    :param k: number of clusters
    :param is_pp: if true, use k-shape++ initialization method.
    :return: mem is the length n array containing the index of the clusters to which
    the series are assigned. centroids is the kxm matrix containing the centroids of
    the clusters
    '''

    # initialization
    n = X.shape[0]
    mem = np.zeros(n)
    if is_pp:
        centroids = kshape_pp_initialization(X,k)
    else:
        initial_centroids = random.sample(range(n), k)
        centroids = X[initial_centroids, :]

    for iter in range(100):
        print(iter)
        prev_mem = mem.copy()
        cluster_cnt = np.zeros(k)
        empty_cluster_cnt = 0
        D = math.inf * np.ones((n, k))
        # assignment
        for i in range(n):
            for j in range(k):
                dist = 1 - max(SINK.NCC(X[i, :], centroids[j, :]))
                D[i, j] = dist

        for i in range(n):
            mem[i] = np.argmin(D[i, :])
            cluster_cnt[int(mem[i])] = cluster_cnt[int(mem[i])] + 1

        if linalg.norm(mem - prev_mem) == 0:
            break

        # check for empty clusters
        empty_cluster_list = []
        for cluster in range(k):
            if cluster_cnt[cluster] == 0:
                empty_cluster_cnt = empty_cluster_cnt + 1
                empty_cluster_list.append(cluster)

        # deal with empty clusters
        if empty_cluster_cnt != 0:
            min_dists = np.amin(D, axis=1)
            templist = np.array(heapq.nlargest(empty_cluster_cnt, enumerate(min_dists), key = lambda x: x[1]))
            distant_points = templist[:, 0]
            for i in range(empty_cluster_cnt):
                mem[int(distant_points[i])] = empty_cluster_list[i]

        # refinement
        for i in range(k):
            centroids[i, :] = kshape_centroid(X, mem, centroids[i, :], i)
            centroids[i, :] = zscore(centroids[i, :])


    return [mem, centroids]


def kshape_pp_initialization(X, k):
    """
    This is based on the k-means++ algorithm. It is an initialization method designed
    to avoid bad initial clusters.
    :param X: Matrix of time series
    :param k: number of clusters
    :return: centroids
    """
    n = X.shape[0]
    centers = np.zeros((k,X.shape[1]))
    ind = random.randrange(k)
    centers[0,:] = X[ind, :]
    for i in range(1,k):
        D = np.ones((n,i)) * np.inf
        weights = np.zeros(n)
        for j in range(n):
            for c in range(i):
                D[j,c] = 1 - max(SINK.NCC(X[j,:], centers[c,:]))
            weights[j] = min(D[j,:])
        ind = random.choices(list(range(n)), weights=weights, k=1)
        centers[i,:] = X[ind,:]
    return centers
