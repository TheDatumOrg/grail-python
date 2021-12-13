from GRAIL.Representation import GRAIL
from GRAIL.TimeSeries import TimeSeries
from GRAIL.kNN import kNN


TRAIN, train_labels = TimeSeries.load("ECG200_TRAIN", "UCR")
TEST, test_labels = TimeSeries.load("ECG200_TEST", "UCR")

representation = GRAIL(kernel="SINK", d = 100, gamma = 5)
repTRAIN, repTEST = representation.get_rep_train_test(TRAIN, TEST, exact=True)
neighbors, _, _ = kNN(repTRAIN, repTRAIN, method="ED", k=5, representation=None,
                              pq_method='opq')

print(neighbors)