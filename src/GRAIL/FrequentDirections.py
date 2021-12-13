#!/usr/bin/python
from numpy import zeros, sqrt, dot, diag
from numpy.linalg import svd, LinAlgError
from scipy.linalg import svd as scipy_svd


class FrequentDirections:
    def __init__(self, d, ell):
        self.d = d
        self.ell = ell
        self.m = 2 * self.ell
        self._sketch = zeros((self.m, self.d))
        self.nextZeroRow = 0

    def append(self, vector):
        if self.nextZeroRow >= self.m:
            self.__rotate__()
        self._sketch[self.nextZeroRow, :] = vector
        self.nextZeroRow += 1

    def __rotate__(self):
        try:
            [_, s, Vt] = svd(self._sketch, full_matrices=False)
        except LinAlgError as err:
            [_, s, Vt] = scipy_svd(self._sketch, full_matrices=False)

        if len(s) >= self.ell:
            sShrunk = sqrt(s[:self.ell] ** 2 - s[self.ell - 1] ** 2)
            self._sketch[:self.ell:, :] = dot(diag(sShrunk), Vt[:self.ell, :])
            self._sketch[self.ell:, :] = 0
            self.nextZeroRow = self.ell
        else:
            self._sketch[:len(s), :] = dot(diag(s), Vt[:len(s), :])
            self._sketch[len(s):, :] = 0
            self.nextZeroRow = len(s)

    def get(self):
        return self._sketch[:self.ell, :]


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, required=True, help='dimension of row vectors (number of columns in matrix).')
    parser.add_argument('-ell', type=int, required=True, help='the number of rows the sketch can keep.')
    args = parser.parse_args()

    fd = FrequentDirections(args.d, args.ell)
    for line in sys.stdin:
        try:
            row = [float(s) for s in line.strip('\n\r').split(',')]
            assert (len(row) == args.d)
        except:
            continue
        fd.append(row)

    for row in fd.get():
        sys.stdout.write('%s\n' % (','.join('%.2E' % x for x in row.flatten())))
