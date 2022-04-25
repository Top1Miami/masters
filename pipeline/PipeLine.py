import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from pipeline.FileReader import FileReader


class PipeLine:
    def __init__(self, directory):
        self.directory = directory

    @staticmethod
    def scale_data(data):
        labels, features = data
        labels.astype(dtype=int)
        scaled_labels = np.where(labels == np.min(labels), 0, 1)
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        return scaled_labels, scaled_features

    def run(self, experiment):
        result_map = {}
        for file_name in os.listdir(self.directory):  # open directory with datasets
            read = FileReader.read_file(self.directory + "/" + file_name)
            if read is None:
                continue
            print(file_name)
            labels, features = PipeLine.scale_data(read)
            labels, features = shuffle(labels, features)
            result_map[file_name[:-4]] = experiment.run(features, labels, file_name)
        return result_map
