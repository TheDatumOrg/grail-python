import csv
import numpy as np

class TimeSeries:
    def __init__(self, time_stamps, values):
        self.time_stamps = time_stamps
        self.values = values


    @classmethod
    def load(self, path, format):
        """
        Loads time series data from a file
        :param path: String. Path to the file that contains time series
        :param format: String. What is the format of the file being loaded from
        :return: A matrix that contains the time series data (numpy array form)
        """
        if format == "UCR":
            return TimeSeries.ucr_load(path)
        elif format == "Single Time Series":
            return TimeSeries.load_single_time_series(path)

    @classmethod
    def load_single_time_series(self, path):
        """
        This is for reading a single time series where every line in the file
        contains first the time stamp then the value
        :param path: path to the file
        :return: TimeSeries object
        """
        with open(path) as csv_file:
            lines = csv.reader(csv_file, delimiter = ',')
            time_stamps = []
            values = []
            for line in lines:
                try:
                    time_stamps.append(int(line[0]))
                    values.append(int(line[1]))
                except ValueError:
                    raise ValueError("There is a problem with the csv file format. Try changing the file encoding to Unicode without BOM.")

            return TimeSeries(time_stamps, values)

    @classmethod
    def ucr_load(self, path, with_time_stamps = False):
        """
        Loads from a csv file in UCR format. Every row contains a time series without timestamps. The first column of every
        row is the class time series belongs to.
        :param path: The file path
        :param with_time_stamps: If this is true then the file starts with explicit time stamps in its first row
        :return: numpy matrix that contains the loaded time series in its rows, an np array of labels of these time series.
        """
        if with_time_stamps == True:
            raise NotImplemented
        labels = []

        with open(path) as csv_file:
            lines = csv.reader(csv_file, delimiter=',')
            ls = []
            for line in lines:
                labels.append(int(line[0]))
                line.pop(0)
                try:
                    ls.append([float(x) for x in line])
                except ValueError:
                    raise ValueError("There is a problem with the csv file format. Try changing the file encoding to Unicode without BOM.")
        return np.array(ls), np.array(labels)




