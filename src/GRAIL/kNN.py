import numpy as np
from GRAIL.Correlation import Correlation
import heapq
import GRAIL.OPQ as OPQ
import GRAIL.PQ as PQ
from time import time

def kNN(TRAIN, TEST, method, k, representation=None, use_exact_rep = False,
        pq_method = None, Ks = 4, M = 16, **kwargs):
    """
    Approximate or exact k-nearest neighbors algorithm depending on the representation
    :param TRAIN: The training set to get the neighbors from
    :param TEST: The test set whose neighbors we get
    :param method: The correlation or distance measure being used
    :param k: how many nearest neighbors to return
    :param representation: The representation being used if any, for instance GRAIL. This is a representation object.
    :param **kwargs: arguments for the correlator
    :return: a matrix of size row(TEST)xk
    """
    if k > TRAIN.shape[0]:
        raise ValueError("Number of nearest neighbors should be smaller than number of points")

    if pq_method:
        return kNN_with_pq(TRAIN, TEST, method, k, representation, use_exact_rep, pq_method, Ks, M, **kwargs)

    rowTEST = TEST.shape[0]
    rowTRAIN = TRAIN.shape[0]

    neighbors = np.zeros((rowTEST, k))
    correlations = np.zeros((rowTEST, k))
    if representation:
        TRAIN, TEST = representation.get_rep_train_test(TRAIN, TEST, exact=use_exact_rep)

    t = time()
    for i in range(rowTEST):
        x = TEST[i, :]
        corr_array = np.zeros(rowTRAIN)
        for j in range(rowTRAIN):
            y = TRAIN[j, :]
            correlation = Correlation(x, y, correlation_protocol_name=method, **kwargs)
            corr_array[j] = correlation.correlate()
        if Correlation.is_similarity(method):
            temp = np.array(heapq.nlargest(k, enumerate(corr_array), key = lambda x: x[1]))
            neighbors[i,:] = temp[:, 0]
            correlations[i,:] = temp[:,1]
        else:
            temp = np.array(heapq.nsmallest(k, enumerate(corr_array), key = lambda x: x[1]))
            neighbors[i,:] = temp[:, 0]
            correlations[i,:] = temp[:,1]
    neighbors = neighbors.astype(int)
    return_time = time() - t
    return neighbors, correlations, return_time

#check the returned distances
def kNN_with_pq(TRAIN, TEST, method, k, representation=None, use_exact_rep = False,
                pq_method = "opq", Ks = 4, M = 16,**kwargs):
    if method != "ED":
        raise ValueError("Product Quantization can only be used with ED.")

    rowTEST = TEST.shape[0]
    rowTRAIN = TRAIN.shape[0]

    neighbors = np.zeros((rowTEST, k))
    distances = np.zeros((rowTEST, k))
    if representation:
        together = np.vstack((TRAIN, TEST))
        if use_exact_rep:
            rep_together = representation.get_exact_representation(together)
        else:
            rep_together = representation.get_representation(together)
        TRAIN = rep_together[0:rowTRAIN, :]
        TEST = rep_together[rowTRAIN:, :]


    if rowTRAIN < Ks:
        Ks = 2 ** int(np.floor(np.log2(rowTRAIN)))

    # This code trims the last parts
    # if TRAIN.shape[1] > M and TRAIN.shape[1] % M != 0:
    #     TRAIN = TRAIN[:, 0:(TRAIN.shape[1] - TRAIN.shape[1] % M)]
    #     TEST = TEST[:, 0:(TRAIN.shape[1] - TRAIN.shape[1] % M)]

    # padding with up to 2^n
    next_pow_2 = int(max(np.ceil(np.log2(TRAIN.shape[1])), np.ceil(np.log2(M))))
    TRAIN = np.hstack((TRAIN, np.zeros((rowTRAIN, 2 ** next_pow_2-TRAIN.shape[1]))))
    TEST = np.hstack((TEST, np.zeros((rowTEST, 2 ** next_pow_2 - TEST.shape[1]))))

    TRAIN = TRAIN.astype(np.float32)
    TEST = TEST.astype(np.float32)



    if pq_method == "opq":
        pq = OPQ.OPQ(M=M, Ks= Ks, verbose=False)
    elif pq_method == "pq":
        pq = PQ.PQ(M=M, Ks= Ks, verbose=False)
    else:
        raise ValueError("Product quantization method not found.")


    pq.fit(vecs=TRAIN)
    TRAIN_code = pq.encode(vecs=TRAIN)
    t = time()

    for i in range(rowTEST):
        query = TEST[i, :]
        dists = pq.dtable(query=query).adist(codes=TRAIN_code)
        temp = np.array(heapq.nsmallest(k, enumerate(dists), key = lambda x: x[1]))
        neighbors[i,:] = temp[:, 0]
        distances[i,:] = temp[:,1]
    neighbors = neighbors.astype(int)
    return_time = time() -t
    #print("Time for M = ", M , ": ", return_time)
    return neighbors, distances, return_time


def kNN_classifier(TRAIN, train_labels, TEST, method, k, representation=None, use_exact_rep = False,
                   pq_method = None, Ks = 4, M = 16, **kwargs):
    neighbors, _, return_time = kNN(TRAIN, TEST, method, k, representation, use_exact_rep, pq_method, Ks, M, **kwargs)
    return_labels = np.zeros(TEST.shape[0])
    for i in range(TEST.shape[0]):
        nearest_labels = np.zeros(k)
        for j in range(k):
            nearest_labels[j] = train_labels[neighbors[i,j]]
        unique, counts = np.unique(nearest_labels, return_counts=True)
        mx = 0
        mx_label = 0
        for j in range(unique.shape[0]):
            if counts[j] > mx:
                mx = counts[j]
                mx_label = unique[j]
        return_labels[i] = mx_label
    return return_labels, return_time

def kNN_classification_precision_test(exact_neighbors, TRAIN, train_labels, TEST, test_labels, method, k, representation=None, use_exact_rep = False,
                   pq_method = None, Ks = 4, M = 16, **kwargs):
    neighbors, _, return_time = kNN(TRAIN, TEST, method, k, representation,
                         use_exact_rep, pq_method, Ks, M, **kwargs)
    return_labels = np.zeros(TEST.shape[0])

    tp_arr = np.zeros(TEST.shape[0])

    for i in range(TEST.shape[0]):
        for j in range(k):
            if neighbors[i,j] in exact_neighbors[i]:
                tp_arr[i] = tp_arr[i] + 1

    precision = np.mean(tp_arr) / k

    #assign labels
    for i in range(TEST.shape[0]):
        nearest_labels = np.zeros(k)
        for j in range(k):
            nearest_labels[j] = train_labels[neighbors[i,j]]
        unique, counts = np.unique(nearest_labels, return_counts=True)
        mx = 0
        mx_label = 0
        for j in range(unique.shape[0]):
            if counts[j] > mx:
                mx = counts[j]
                mx_label = unique[j]
        return_labels[i] = mx_label

    # accuracy of the labels
    cnt_acc = 0

    for i in range(test_labels.shape[0]):
        if test_labels[i] == return_labels[i]:
            cnt_acc += 1
    classification_accuracy = cnt_acc / test_labels.shape[0]
    return classification_accuracy, precision, return_time

#Stands for mean average precision
def MAP(exact_neighbors, neighbors):
    n = exact_neighbors.shape[0]
    k = exact_neighbors.shape[1]
    map_measure = 0
    for i in range(n):
        AP = 0
        psum = 0
        for r in range(k):
            if neighbors[i,r] in exact_neighbors[i]:
                rel = 1
            else:
                rel = 0
            psum += rel
            AP += (psum / (r+1))*rel
        AP /= k
        map_measure += AP
    map_measure /= n
    return map_measure

def avg_recall_measure(exact_neighbors, neighbors):
    n = exact_neighbors.shape[0]
    k = exact_neighbors.shape[1]

    avg_recall = 0
    for i in range(n):
        recall = 0
        for r in range(k):
            if neighbors[i,r] in exact_neighbors[i]:
                recall += 1
        recall /= k
        avg_recall += recall
    avg_recall /= n
    return avg_recall
