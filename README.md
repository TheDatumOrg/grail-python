# GRAIL

A Python implementation of [GRAIL](http://people.cs.uchicago.edu/~jopa/Papers/PaparrizosVLDB2019.pdf), a generic framework to learn compact time series representations. 

## Requirements

- Python 3.6+
- `numpy`
- `scipy`
- `tslearn`

## Installation

Installation using pip:

`pip install grailts`

To install from the source:

`python setup.py install`

## Usage

### Full Example

Here is an example where we load a [UCR](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) dataset and run approximate k-nearest neighbors on its GRAIL representations:

```python
from GRAIL.TimeSeries import TimeSeries
from GRAIL.Representation import GRAIL
from GRAIL.kNN import kNN

TRAIN, train_labels = TimeSeries.load("ECG200_TRAIN", "UCR")
TEST, test_labels = TimeSeries.load("ECG200_TEST", "UCR")

representation = GRAIL(kernel="SINK", d = 100, gamma = 5)
repTRAIN, repTEST = representation.get_rep_train_test(TRAIN, TEST, exact=True)
neighbors, _, _ = kNN(repTRAIN, repTEST, method="ED", k=5, representation=None,
                              pq_method='opq')

print(neighbors)
```

### Loading Datasets

To load UCR type datasets:

```python
TRAIN, train_labels = TimeSeries.load("ECG200_TRAIN", "UCR")
TEST, test_labels = TimeSeries.load("ECG200_TEST", "UCR")
```

In this package, we assume that each row of the datasets is a time series. 

### Fetch GRAIL Representations

To fetch exact GRAIL representations of a training and a test dataset:

```python
representation = GRAIL(kernel="SINK", d = 100, gamma = 5)
repTRAIN, repTEST = representation.get_rep_train_test(TRAIN, TEST, exact=True)
```

Here `d` specifies the number of landmark series, and `gamma` specifies the hyperparameter used for the SINK kernel. If `gamma` is not specified, it will be tuned by the algorithm. 

If a single dataset is used instead:

`repX = representation.get_representation(X)`

### Get Approximate k-Nearest-Neighbors

To get the approximate k-Nearest-Neighbors of `TEST` in `TRAIN` use:

```python
neighbors, correlations, return_time = kNN(repTRAIN, repTEST, method="ED", k=5, representation=None,
                              pq_method='opq')
```

Note that Euclidean Distance in the GRAIL representation space estimates the SINK correlation in the original space. 